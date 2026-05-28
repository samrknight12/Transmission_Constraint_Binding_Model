"""
Layer 1 SHAP-based explainability for MISO flowgate binding models.

Uses XGBoost's native pred_contribs (equivalent to TreeSHAP) to avoid
version incompatibilities with the shap library.

Parts:
    1. Global importance  — per-flowgate mean |SHAP| on test set
    2. explain_prediction — single-hour explanation dict
    3. generate_narrative — plain-English string for traders
    4. explain_day        — full-day batch explanation DataFrame

Usage:
    # Part 1 — compute and save global importance for all production models
    py -3 src/explainability/shap_explainer.py

    # Part 4 — run a specific day
    py -3 src/explainability/shap_explainer.py --day 2024-10-15
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.train import (
    TARGET_COL,
    RESULTS_DIR,
    _safe_id,
    _parquet_path,
)
from scripts.retrain_seasonal_features import add_seasonal_features

# ── Constants ───────────────────────────────────────────────────────────────────

MODELS_DIR   = Path("models/saved")
SHAP_DIR     = Path("models/shap/global")
DAILY_DIR    = Path("data/results/daily_explanations")
EVAL_CSV     = RESULTS_DIR / "test_evaluation.csv"
TRAIN_CSV    = RESULTS_DIR / "training_results.csv"

_TEST_START = pd.Timestamp("2024-10-01 05:00", tz="UTC")
_TEST_END   = pd.Timestamp("2025-01-01 04:00", tz="UTC")

# ── Narrative exclusion list ─────────────────────────────────────────────────────

# Features that influence the model but should not be surfaced in narrative text.
# These are autocorrelation / look-back signals that are real but not actionable
# for a trader reading a morning brief.
EXCLUDED_FROM_NARRATIVE: list[str] = [
    "hours_since_last_binding",
    "binding_freq_trailing_7d",
    "binding_freq_trailing_30d",
]

# Internal mapping: substrings matched against driver["feature_name"].
# "binding_freq_" (no time suffix) covers 7d / 30d / 90d — all share the same
# label and the same exclusion rationale, so all windows are suppressed together.
_NARRATIVE_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    "hours_since_binding",    # flowgate_hours_since_binding
    "days_since_last_binding", # days_since_last_binding_rolling_14d
    "binding_freq_",          # flowgate_binding_freq_7d / 30d / 90d
})


def _is_excluded_from_narrative(driver: dict) -> bool:
    feat = driver["feature_name"]
    return any(p in feat for p in _NARRATIVE_EXCLUDE_PATTERNS)


# ── Calendar feature suppression ─────────────────────────────────────────────────

# Calendar / encoded features that should never appear with raw numeric values
# in narrative text. If one is the top driver, it is replaced with the generic
# phrase "seasonal and time-of-day patterns". Subsequent calendar drivers are
# silently skipped so the next actionable driver takes their slot.
SUPPRESS_FROM_NARRATIVE: list[str] = [
    "day_of_week", "month_of_year", "month",
    "hour_sin", "hour_cos", "season",
    "is_weekend", "is_holiday",
]

_CALENDAR_FEATURE_NAMES: frozenset[str] = frozenset({
    "hour_of_day", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "month", "month_sin", "month_cos",
    "season",
    "is_weekend", "is_nerc_holiday",
    "is_peak_hour", "is_shoulder_hour", "is_peak_period",
})

_CALENDAR_LABELS: frozenset[str] = frozenset({
    "time of day", "day of week", "month of year",
    "season", "weekend flag", "NERC holiday",
    "peak period", "shoulder period",
})


def _is_calendar_driver(driver: dict) -> bool:
    return (
        driver["feature_name"] in _CALENDAR_FEATURE_NAMES
        or driver["label"] in _CALENDAR_LABELS
    )


# ── Feature label mapping ────────────────────────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "wind_forecast_mw":              "wind generation forecast",
    "wind_ahead_1h_mw":              "wind ramp rate",
    "wind_ahead_4h_mw":              "wind ramp rate",
    "thermal_outage_mw":             "thermal outage level",
    "outage_mw_total":               "thermal outage level",
    "planned_outage_mw":             "planned outage MW",
    "forced_outage_mw":              "forced outage MW",
    "outage_mw_ptdf_weighted":       "PTDF-weighted outage MW",
    "outage_pct_of_capacity":        "outage share of capacity",
    "is_outage_proxy":               "outage proxy flag",
    "load_forecast_mw":              "system load forecast",
    "load_forecast_ahead_1h_mw":     "load ramp rate",
    "load_forecast_ahead_4h_mw":     "load ramp rate",
    "load_pct_of_peak":              "load as share of peak",
    "load_change_1h_mw":             "load ramp rate",
    "load_change_4h_mw":             "load ramp rate",
    "load_change_24h_mw":            "load ramp rate",
    "load_ramp_mw_per_hour":         "load ramp rate",
    "load_deviation_from_avg_mw":    "load deviation from average",
    "load_deviation_from_7d_mean":   "load deviation from average",
    "flowgate_loading_pct":          "current line loading",
    "flowgate_loading_chg_1h":       "line loading change",
    "flowgate_loading_chg_4h":       "line loading change",
    "flowgate_loading_chg_24h":      "line loading change",
    "flowgate_distance_to_limit":    "distance to thermal limit",
    "flowgate_pct_of_30d_max":       "loading vs 30-day max",
    "flowgate_binding_freq_7d":      "recent binding frequency",
    "flowgate_binding_freq_30d":     "recent binding frequency",
    "flowgate_binding_freq_90d":     "recent binding frequency",
    "flowgate_hours_since_binding":  "hours since last binding",
    "binding_rate_same_month_prior_year":  "prior-year same-month binding rate",
    "days_since_last_binding_rolling_14d": "days since last binding",
    "solar_forecast_mw":             "solar generation forecast",
    "solar_ramp_1h_mw":              "solar ramp rate",
    "renewable_total_mw":            "renewable generation total",
    "renewable_penetration_pct":     "renewable penetration",
    "wind_ramp_1h_mw":               "wind ramp rate",
    "wind_ramp_4h_mw":               "wind ramp rate",
    "wind_variability_4h_mw":        "wind variability",
    "wind_forecast_rolling_mae_7d":  "wind forecast error",
    "thermal_da_cleared_mw":         "DA cleared thermal generation",
    "thermal_rt_actual_mw":          "RT actual thermal generation",
    "thermal_deviation_30d_mw":      "thermal deviation from average",
    "thermal_rt_vs_da_gap_mw":       "thermal DA/RT deviation",
    "outage_mw_change_24h":          "outage change last 24 hours",
    "forced_outage_fraction":        "forced outage share",
    "outage_count":                  "number of outages",
    "hour_sin":                      "time of day",
    "hour_cos":                      "time of day",
    "hour_of_day":                   "time of day",
    "day_of_week":                   "day of week",
    "dow_sin":                       "day of week",
    "dow_cos":                       "day of week",
    "month":                         "month of year",
    "month_sin":                     "month of year",
    "month_cos":                     "month of year",
    "season":                        "season",
    "is_weekend":                    "weekend flag",
    "is_peak_hour":                  "peak period",
    "is_shoulder_hour":              "shoulder period",
    "is_nerc_holiday":               "NERC holiday",
    "is_peak_period":                "peak period",
    "flowgate_loading_pct_is_observed": "line loading observed flag",
}


def _label(feature_name: str) -> str:
    if feature_name in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature_name]
    return re.sub(r"_+", " ", feature_name).strip()


# ── Safe probability prediction ─────────────────────────────────────────────────

def safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Predict class-1 probability and clip to (0.001, 0.999)."""
    proba = model.predict_proba(X)[:, 1]
    return np.clip(proba, 0.001, 0.999)


# ── Model loading ────────────────────────────────────────────────────────────────

def _load_model(flowgate_id: str) -> tuple:
    """
    Returns (base_model, calibrator_or_None, uses_calibration: bool).
    Prefers calibrated joblib when it exists.
    """
    safe = _safe_id(flowgate_id)
    calib_path = MODELS_DIR / f"{safe}_calibrated.joblib"
    base_path  = MODELS_DIR / f"{safe}.joblib"

    if calib_path.exists():
        bundle = joblib.load(calib_path)
        return bundle["base_model"], bundle["calibrator"], True

    if base_path.exists():
        m = joblib.load(base_path)
        return m, None, False

    raise FileNotFoundError(f"No model found for {flowgate_id}")


def _get_features(flowgate_id: str, base_model) -> list[str]:
    return list(base_model.feature_names_in_)


# ── SHAP via XGBoost native pred_contribs ───────────────────────────────────────

def _shap_values(base_model, X: pd.DataFrame) -> np.ndarray:
    """
    Returns array (n_samples, n_features) of log-odds SHAP contributions.
    Last column from pred_contribs is the bias term — dropped here.
    """
    dmat = xgb.DMatrix(X)
    contribs = base_model.get_booster().predict(dmat, pred_contribs=True)
    return contribs[:, :-1]  # drop bias column


def _load_test_data(flowgate_id: str, features: list[str]) -> pd.DataFrame:
    """Load parquet and filter to test window. Adds seasonal features if needed."""
    df = pd.read_parquet(_parquet_path(flowgate_id))
    needs_seasonal = (
        "binding_rate_same_month_prior_year" in features
        or "days_since_last_binding_rolling_14d" in features
    )
    if needs_seasonal:
        df = add_seasonal_features(df)
    mask = (df.index >= _TEST_START) & (df.index <= _TEST_END)
    return df.loc[mask]


# ── Part 1: Global importance ────────────────────────────────────────────────────

def compute_global_importance(flowgate_id: str) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature on the test window.
    Returns DataFrame with columns [feature, mean_abs_shap] sorted descending.
    """
    base_model, _, _ = _load_model(flowgate_id)
    features = _get_features(flowgate_id, base_model)
    test_df  = _load_test_data(flowgate_id, features)
    X_test   = test_df[features]

    sv = _shap_values(base_model, X_test)
    mean_abs = np.abs(sv).mean(axis=0)

    importance = pd.DataFrame({
        "feature":       features,
        "mean_abs_shap": mean_abs,
        "label":         [_label(f) for f in features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance["rank"] = importance.index + 1
    return importance


def save_global_importance(flowgate_id: str) -> Path:
    importance = compute_global_importance(flowgate_id)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    out = SHAP_DIR / f"{_safe_id(flowgate_id)}_importance.csv"
    importance.to_csv(out, index=False)
    return out


def run_global_importance_all(verbose: bool = True) -> None:
    """Part 1 entry point: compute and save importance for all production models."""
    train_df = pd.read_csv(TRAIN_CSV)
    prod_ids = train_df[train_df["model_status"] == "production"]["flowgate_id"].tolist()

    errors: list[str] = []
    for i, fg_id in enumerate(prod_ids, 1):
        if verbose:
            print(f"[{i:>2}/{len(prod_ids)}] {fg_id}", end="  ", flush=True)
        try:
            out = save_global_importance(fg_id)
            if verbose:
                imp = pd.read_csv(out)
                top = imp.iloc[0]
                print(f"top={top['feature']} ({top['mean_abs_shap']:.4f})")
        except Exception as exc:
            errors.append(f"{fg_id}: {exc}")
            if verbose:
                print(f"ERROR: {exc}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    elif verbose:
        print(f"\nSaved {len(prod_ids)} importance files to {SHAP_DIR}/")


# ── Part 2: Single prediction explanation ────────────────────────────────────────

def _value_context(feature: str, value: float, df_full: pd.DataFrame) -> str:
    """Build a short context string for a feature value (e.g. 'X MW -- Y% above avg')."""
    col_mean = df_full[feature].mean() if feature in df_full.columns else None

    if pd.isna(value):
        return "N/A"

    # Binary flags
    if feature.startswith("is_") or feature == "flowgate_loading_pct_is_observed":
        return "yes" if value >= 0.5 else "no"

    # Hours / days since last event
    if "hours_since" in feature:
        return f"{value:.0f} hrs"
    if "days_since" in feature:
        return f"{value:.1f} days"

    # Calendar / cyclic encoding features — show raw integer, no deviation framing
    _CALENDAR_FEATURES = {
        "hour_of_day", "day_of_week", "month", "season",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    }
    if feature in _CALENDAR_FEATURES:
        return str(int(round(value)))

    # MW-scale features: show absolute value + deviation from mean
    _MW_FEATURES = {
        "wind_forecast_mw",
        "thermal_outage_mw", "outage_mw_total", "load_forecast_mw",
        "solar_forecast_mw", "renewable_total_mw",
        "planned_outage_mw", "forced_outage_mw", "outage_mw_ptdf_weighted",
        "thermal_da_cleared_mw", "thermal_rt_actual_mw",
    }
    if feature in _MW_FEATURES and col_mean is not None and abs(col_mean) > 1:
        pct_dev = (value - col_mean) / abs(col_mean) * 100
        direction = "above" if pct_dev >= 0 else "below"
        return f"{value:,.0f} MW -- {abs(pct_dev):.0f}% {direction} avg"

    # Signed MW delta / ramp features
    _RAMP_FEATURES = {
        "wind_ramp_1h_mw", "wind_ramp_4h_mw", "wind_ahead_1h_mw", "wind_ahead_4h_mw",
        "solar_ramp_1h_mw",
        "load_change_1h_mw", "load_change_4h_mw", "load_change_24h_mw",
        "load_forecast_ahead_1h_mw", "load_forecast_ahead_4h_mw",
        "load_deviation_from_avg_mw", "load_deviation_from_7d_mean",
        "wind_variability_4h_mw", "wind_forecast_rolling_mae_7d",
        "outage_mw_change_24h", "thermal_deviation_30d_mw", "thermal_rt_vs_da_gap_mw",
    }
    if feature in _RAMP_FEATURES:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:,.0f} MW"

    # Binding frequency features (stored as 0-1 ratio, display as percent)
    _RATIO_FEATURES = {
        "flowgate_binding_freq_7d", "flowgate_binding_freq_30d",
        "flowgate_binding_freq_90d", "flowgate_pct_of_30d_max",
        "load_pct_of_peak", "forced_outage_fraction",
        "binding_rate_same_month_prior_year",
    }
    if feature in _RATIO_FEATURES:
        return f"{value * 100:.1f}%"

    # Features already stored in percent points (0-100 range)
    _PCT_POINT_FEATURES = {
        "renewable_penetration_pct", "flowgate_loading_pct",
        "outage_pct_of_capacity",
    }
    if feature in _PCT_POINT_FEATURES:
        if col_mean is not None and abs(col_mean) > 0.1:
            dev = value - col_mean
            direction = "above" if dev >= 0 else "below"
            return f"{value:.1f}% -- {abs(dev):.1f}pp {direction} avg"
        return f"{value:.1f}%"

    # Flowgate loading change (pp)
    if "flowgate_loading_chg" in feature:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}pp"

    # Distance to limit (already in headroom units)
    if "distance_to_limit" in feature:
        return f"{value:.1f}"

    # Fallback: show value and deviation from mean if available
    if col_mean is not None and abs(col_mean) > 0:
        pct_dev = (value - col_mean) / abs(col_mean) * 100
        direction = "above" if pct_dev >= 0 else "below"
        return f"{value:.3g} ({abs(pct_dev):.0f}% {direction} avg)"

    return f"{value:.3g}"


_DIRECTION_WARNINGS_CSV = RESULTS_DIR / "shap_direction_warnings.csv"
_DIRECTION_WARNING_COLS = [
    "logged_utc", "flowgate_id", "datetime_utc",
    "predicted_probability", "top_feature", "shap_value",
    "feature_value", "col_mean", "col_std",
]


def _log_direction_warning(
    flowgate_id: str,
    ts: pd.Timestamp,
    prob: float,
    top_driver: dict,
    col_mean: float,
    col_std: float,
) -> None:
    import csv
    from datetime import datetime, timezone

    row = {
        "logged_utc":            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "flowgate_id":           flowgate_id,
        "datetime_utc":          str(ts),
        "predicted_probability": f"{prob:.4f}",
        "top_feature":           top_driver["feature_name"],
        "shap_value":            f"{top_driver['shap_value']:.4f}",
        "feature_value":         f"{top_driver['feature_value']:.4f}",
        "col_mean":              f"{col_mean:.4f}",
        "col_std":               f"{col_std:.4f}",
    }
    _DIRECTION_WARNINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _DIRECTION_WARNINGS_CSV.exists()
    with open(_DIRECTION_WARNINGS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_DIRECTION_WARNING_COLS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def explain_prediction(flowgate_id: str, datetime_utc: str | pd.Timestamp) -> dict:
    """
    Explain the model prediction for a single flowgate-hour.

    Returns:
        {
          predicted_probability: float,
          top_drivers: [
              {feature_name, label, shap_value, feature_value, feature_value_context},
              ...  (up to 8, sorted by |shap_value| desc, deduped by label)
          ],
          binding_base_rate: float,
          prediction_context: "above_base" | "below_base",
          direction_suspect: bool,  -- True when top SHAP driver is "downward" yet
                                       prob > 0.85 and the feature is well above average;
                                       indicates a non-linear interaction the narrative
                                       may be misrepresenting
        }
    """
    ts = pd.Timestamp(datetime_utc, tz="UTC") if not isinstance(datetime_utc, pd.Timestamp) else datetime_utc

    base_model, calibrator, uses_calibration = _load_model(flowgate_id)
    features = _get_features(flowgate_id, base_model)

    df_full = pd.read_parquet(_parquet_path(flowgate_id))
    needs_seasonal = (
        "binding_rate_same_month_prior_year" in features
        or "days_since_last_binding_rolling_14d" in features
    )
    if needs_seasonal:
        df_full = add_seasonal_features(df_full)

    if ts not in df_full.index:
        raise KeyError(f"{ts} not found in feature data for {flowgate_id}")

    row = df_full.loc[[ts], features]

    # Predicted probability
    raw_prob = float(safe_predict_proba(base_model, row)[0])
    if calibrator is not None:
        prob = float(np.clip(calibrator.predict(np.array([[raw_prob]]))[0], 0.001, 0.999))
    else:
        prob = raw_prob

    # SHAP values for this row
    sv = _shap_values(base_model, row)[0]  # shape (n_features,)

    # Binding base rate from training window
    train_mask = df_full.index < _TEST_START
    base_rate = float(df_full.loc[train_mask, TARGET_COL].mean()) if train_mask.any() else float("nan")

    # Top drivers by |shap|
    ranked = np.argsort(np.abs(sv))[::-1]
    seen_labels: set[str] = set()
    top_drivers = []
    for idx in ranked:
        feat = features[idx]
        lbl  = _label(feat)
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        feat_val = float(row.iloc[0][feat])
        top_drivers.append({
            "feature_name":          feat,
            "label":                 lbl,
            "shap_value":            float(sv[idx]),
            "feature_value":         feat_val,
            "feature_value_context": _value_context(feat, feat_val, df_full),
        })
        if len(top_drivers) == 8:
            break

    # Direction-suspect check: high-confidence prediction (>0.85) where the top
    # SHAP driver points downward yet the feature value is substantially above
    # its historical mean. This indicates a non-linear interaction at a boundary
    # where the local SHAP gradient reverses — the narrative direction word would
    # be misleading. Log for review; generate_narrative() swaps the direction word.
    direction_suspect = False
    if prob > 0.85 and top_drivers and top_drivers[0]["shap_value"] < 0:
        td = top_drivers[0]
        feat = td["feature_name"]
        val  = td["feature_value"]
        if feat in df_full.columns:
            col_mean = float(df_full[feat].mean())
            col_std  = float(df_full[feat].std())
            threshold = col_mean + max(col_std, abs(col_mean) * 0.5)
            if col_std > 0 and val > threshold:
                direction_suspect = True
                _log_direction_warning(flowgate_id, ts, prob, td, col_mean, col_std)

    return {
        "predicted_probability": prob,
        "top_drivers":           top_drivers,
        "binding_base_rate":     base_rate,
        "prediction_context":    "above_base" if prob > base_rate else "below_base",
        "direction_suspect":     direction_suspect,
    }


# ── Part 3: Narrative generator ──────────────────────────────────────────────────

def generate_narrative(flowgate_id: str, datetime_utc: str | pd.Timestamp) -> str:
    """
    Return a plain-English 3-sentence narrative for a power trader.
    No references to SHAP, ML, or model internals.
    """
    ts  = pd.Timestamp(datetime_utc, tz="UTC") if not isinstance(datetime_utc, pd.Timestamp) else datetime_utc
    exp = explain_prediction(flowgate_id, ts)

    prob             = exp["predicted_probability"]
    base_rate        = exp["binding_base_rate"]
    context          = exp["prediction_context"]
    drivers          = exp["top_drivers"]
    direction_suspect = exp["direction_suspect"]

    prob_pct  = prob * 100
    base_pct  = base_rate * 100
    hour_str  = ts.strftime("%H:%M UTC")
    date_str  = ts.strftime("%Y-%m-%d")

    # Sentence 1: probability and direction vs base rate
    if context == "above_base":
        diff = prob_pct - base_pct
        s1 = (
            f"At {hour_str} on {date_str}, {flowgate_id} has a {prob_pct:.0f}% binding "
            f"probability -- {diff:.0f} percentage points above its historical base rate of "
            f"{base_pct:.0f}%."
        )
    else:
        diff = base_pct - prob_pct
        s1 = (
            f"At {hour_str} on {date_str}, {flowgate_id} has a {prob_pct:.0f}% binding "
            f"probability -- {diff:.0f} percentage points below its historical base rate of "
            f"{base_pct:.0f}%."
        )

    # Filter out binding-history features — real signal but not actionable for traders.
    # Exception: if every driver is excluded, fall back to a generic phrase.
    narrative_drivers = [d for d in drivers if not _is_excluded_from_narrative(d)]
    all_excluded_fallback = len(narrative_drivers) == 0

    # Further partition: calendar features (encoded cyclics) are also not readable
    # in raw form. Track whether the top narrative driver is a calendar feature so
    # we can substitute "seasonal and time-of-day patterns" as the primary phrase
    # and skip to the next actionable driver for context sentences.
    calendar_was_top = (
        not all_excluded_fallback
        and len(narrative_drivers) > 0
        and _is_calendar_driver(narrative_drivers[0])
    )
    non_calendar_drivers = [d for d in narrative_drivers if not _is_calendar_driver(d)]
    all_calendar_fallback = not all_excluded_fallback and len(non_calendar_drivers) == 0

    d1 = non_calendar_drivers[0] if len(non_calendar_drivers) > 0 else None
    d2 = non_calendar_drivers[1] if len(non_calendar_drivers) > 1 else None
    d3 = non_calendar_drivers[2] if len(non_calendar_drivers) > 2 else None

    # Sentence 2: primary and secondary driver
    if all_excluded_fallback:
        direction_word = "upward" if context == "above_base" else "downward"
        s2 = f"Recent binding activity is the primary driver, pushing probability {direction_word}."
    elif all_calendar_fallback:
        direction_word = "upward" if context == "above_base" else "downward"
        s2 = (
            f"Seasonal and time-of-day patterns are the primary driver, "
            f"pushing probability {direction_word}."
        )
    elif calendar_was_top and d1:
        # Calendar was top; lead with generic phrase, then name the first non-calendar driver.
        # direction_suspect swap applies to d1 if it was also the absolute top driver.
        direction_1 = "upward" if (d1["shap_value"] > 0 or direction_suspect) else "downward"
        s2 = (
            f"Seasonal and time-of-day patterns are the primary driver; "
            f"followed by {d1['label']} ({d1['feature_value_context']}) pulling {direction_1}."
        )
    elif d1:
        # direction_suspect: top absolute SHAP driver has downward sign but high feature
        # value — narrative direction swapped to "upward" to reflect likely true influence.
        if direction_suspect and d1["shap_value"] < 0:
            direction_1 = "upward"
        else:
            direction_1 = "upward" if d1["shap_value"] > 0 else "downward"
        s2_parts = [
            f"{d1['label']} ({d1['feature_value_context']}) is the primary driver, "
            f"pushing probability {direction_1}"
        ]
        if d2:
            direction_2 = "upward" if d2["shap_value"] > 0 else "downward"
            s2_parts.append(
                f"followed by {d2['label']} ({d2['feature_value_context']}) pulling {direction_2}"
            )
        s2 = "; ".join(s2_parts) + "."
    else:
        s2 = "No dominant drivers identified."

    # Sentence 3: third driver or summary
    # When calendar was top, d1/d2 are shifted by one vs the standard case,
    # so use d2 for s3 to avoid repeating the d1 just used in s2.
    s3_driver = d2 if calendar_was_top else d3
    if not all_excluded_fallback and not all_calendar_fallback and s3_driver:
        direction_3 = "elevated" if s3_driver["shap_value"] > 0 else "suppressed"
        s3 = (
            f"Additionally, {s3_driver['label']} ({s3_driver['feature_value_context']}) "
            f"contributes to {direction_3} risk."
        )
    else:
        summary = "elevated" if context == "above_base" else "reduced"
        s3 = f"Overall conditions point to {summary} congestion risk for this constraint."

    return f"{s1} {s2} {s3}"


# ── Consecutive-run deduplication ───────────────────────────────────────────────

def _dedup_consecutive_runs(narratives: list[dict]) -> list[dict]:
    """
    Collapse runs of 3+ consecutive hours where the same flowgate has identical
    probability (rounded to 3 dp) into a single row.

    Input list must already be sorted by (datetime, -probability).

    Collapsed row:
      datetime   → "HH:MM-HH:MM UTC"  (first to last hour of the run)
      probability → probability of the first hour (all are equal by definition)
      narrative  → first hour's narrative + suffix noting the run length
    """
    # Group rows by flowgate_id, preserving original list order for non-grouped rows
    by_fg: dict[str, list[dict]] = {}
    for row in narratives:
        by_fg.setdefault(row["flowgate_id"], []).append(row)

    # Process each flowgate independently, then re-merge
    processed: dict[str, list[dict]] = {}
    for fg_id, fg_rows in by_fg.items():
        out: list[dict] = []
        i = 0
        while i < len(fg_rows):
            prob_key = round(fg_rows[i]["probability"], 3)
            run = [fg_rows[i]]

            # Extend while next hour is calendar-consecutive and same rounded prob
            while i + len(run) < len(fg_rows):
                nxt = fg_rows[i + len(run)]
                prev_ts = pd.Timestamp(run[-1]["datetime"])
                curr_ts = pd.Timestamp(nxt["datetime"])
                if (curr_ts - prev_ts == pd.Timedelta(hours=1)
                        and round(nxt["probability"], 3) == prob_key):
                    run.append(nxt)
                else:
                    break

            if len(run) >= 3:
                first_ts = pd.Timestamp(run[0]["datetime"])
                last_ts  = pd.Timestamp(run[-1]["datetime"])
                dt_str   = f"{first_ts.strftime('%H:%M')}-{last_ts.strftime('%H:%M')} UTC"
                suffix   = (
                    f" [Probability unchanged for {len(run)} consecutive hours"
                    f" -- synthetic loading proxy active]"
                )
                out.append({
                    "datetime":    dt_str,
                    "flowgate_id": fg_id,
                    "probability": run[0]["probability"],
                    "narrative":   run[0]["narrative"] + suffix,
                })
            else:
                out.extend(run)

            i += len(run)

        processed[fg_id] = out

    # Re-interleave: iterate over original list order, emit each fg's processed rows
    # in order, skipping already-emitted ones.
    emitted: dict[str, int] = {fg: 0 for fg in processed}
    result: list[dict] = []
    seen_fgs: list[str] = []
    for row in narratives:
        fg = row["flowgate_id"]
        if fg not in seen_fgs:
            seen_fgs.append(fg)
    for fg in seen_fgs:
        result.extend(processed[fg])

    # Re-sort: collapsed rows have string datetimes, so sort key uses the first
    # time token (HH:MM) which is still lexicographically correct within a day.
    def _sort_key(r: dict) -> tuple:
        dt = r["datetime"]
        if isinstance(dt, str):
            time_part = dt.split("-")[0].replace(" UTC", "")  # "HH:MM"
        else:
            time_part = pd.Timestamp(dt).strftime("%H:%M")
        return (time_part, -r["probability"])

    result.sort(key=_sort_key)
    return result


# ── Market summary ───────────────────────────────────────────────────────────────

def _extract_pct_range(narratives_df: pd.DataFrame, label: str) -> str:
    """
    Scan narrative text for '{label} (... N% above/below avg ...)' patterns.
    Return a human-readable range string like '279-346% above average',
    or an empty string if no parseable percentages found.
    """
    import re
    pct_re = re.compile(
        re.escape(label) + r"\s*\([^)]*?(\d[\d,]*)%\s+(above|below)\s+avg[^)]*\)",
        re.IGNORECASE,
    )
    pcts_above: list[float] = []
    pcts_below: list[float] = []
    for narr in narratives_df["narrative"].dropna():
        for m in pct_re.finditer(narr):
            val = float(m.group(1).replace(",", ""))
            if m.group(2).lower() == "above":
                pcts_above.append(val)
            else:
                pcts_below.append(val)

    if pcts_above:
        lo, hi = int(min(pcts_above)), int(max(pcts_above))
        rng = f"{lo}%" if lo == hi else f"{lo}-{hi}%"
        return f"{rng} above average"
    if pcts_below:
        lo, hi = int(min(pcts_below)), int(max(pcts_below))
        rng = f"{lo}%" if lo == hi else f"{lo}-{hi}%"
        return f"{rng} below average"
    return ""


def _fg_short_name(fg_id: str) -> str:
    """Return the substation portion before ' FLO', or the full id if absent."""
    return fg_id.split(" FLO")[0] if " FLO" in fg_id else fg_id


def generate_market_summary(narratives_df: pd.DataFrame, target_date: "date") -> str:
    """
    Derive a 2-3 sentence market summary from the day's narratives.

    Counts which readable feature labels appear across flowgates.
    Labels present in 3+ distinct flowgate narratives are 'dominant drivers'.
    """
    from collections import defaultdict

    n_constraints = narratives_df["flowgate_id"].nunique()
    date_header = f"{target_date.strftime('%B')} {target_date.day}"

    # ── Extract labels per flowgate ──────────────────────────────────────────────
    # Sort labels longest-first so "solar generation forecast" matches before "solar"
    unique_labels = sorted(set(_FEATURE_LABELS.values()), key=len, reverse=True)

    fg_labels: dict[str, set[str]] = defaultdict(set)
    for _, row in narratives_df.iterrows():
        narr = row["narrative"].lower()
        fg   = row["flowgate_id"]
        for label in unique_labels:
            if label.lower() in narr:
                fg_labels[fg].add(label)

    # ── Dominant drivers: label appears in 3+ distinct flowgates ────────────────
    label_fg_map: dict[str, list[str]] = defaultdict(list)
    for fg, labels in fg_labels.items():
        for label in labels:
            label_fg_map[label].append(fg)

    dominant = sorted(
        [(label, fgs) for label, fgs in label_fg_map.items() if len(set(fgs)) >= 3],
        key=lambda x: len(set(x[1])),
        reverse=True,
    )

    # ── Most persistent flowgate (most output rows) ──────────────────────────────
    fg_counts = narratives_df["flowgate_id"].value_counts()
    top_fg       = fg_counts.index[0] if not fg_counts.empty else None
    top_fg_count = int(fg_counts.iloc[0]) if not fg_counts.empty else 0

    # ── Sentence 1: overview ─────────────────────────────────────────────────────
    s1 = (
        f"{date_header} congestion summary: "
        f"{n_constraints} constraint{'s' if n_constraints != 1 else ''} "
        f"flagged above threshold."
    )

    # ── Sentence 2: dominant driver(s) ──────────────────────────────────────────
    if dominant:
        top_label, top_fgs = dominant[0]
        unique_top_fgs = list(dict.fromkeys(top_fgs))  # preserve order, deduplicate

        pct_range = _extract_pct_range(narratives_df, top_label)
        context_part = f" ({pct_range})" if pct_range else ""

        # Name up to 2 flowgates compactly
        fg_names = " and ".join(_fg_short_name(f) for f in unique_top_fgs[:2])
        if len(unique_top_fgs) > 2:
            fg_names += f" (+{len(unique_top_fgs) - 2} others)"

        s2 = (
            f"{top_label.capitalize()}{context_part} is the dominant driver "
            f"across {fg_names}."
        )

        # Mention a second dominant driver if one exists
        if len(dominant) > 1:
            second_label, second_fgs = dominant[1]
            n_second = len(set(second_fgs))
            s2 += (
                f" {second_label.capitalize()} also contributes "
                f"across {n_second} constraint{'s' if n_second != 1 else ''}."
            )
    else:
        s2 = "No single driver dominates across 3 or more constraints today."

    # ── Sentence 3: most persistent constraint ───────────────────────────────────
    if top_fg and top_fg_count >= 2:
        fg_short = _fg_short_name(top_fg)
        # Find this flowgate's most-mentioned non-generic driver
        fg_narrs = narratives_df[narratives_df["flowgate_id"] == top_fg]["narrative"]
        driver_counts: dict[str, int] = defaultdict(int)
        for narr in fg_narrs:
            for label in unique_labels:
                if label.lower() in narr.lower():
                    driver_counts[label] += 1
        # Pick the top driver that isn't a generic fallback phrase
        top_driver = max(driver_counts, key=driver_counts.get) if driver_counts else None
        driver_phrase = f" driven by {top_driver}" if top_driver else ""
        s3 = f"{fg_short} remains elevated across {top_fg_count} output row{'s' if top_fg_count != 1 else ''}{driver_phrase}."
    else:
        s3 = "Elevated risk is distributed with no single constraint persistently dominant."

    return f"{s1} {s2} {s3}"


# ── Part 4: Batch daily explanation ─────────────────────────────────────────────

def explain_day(
    target_date: str | date,
    top_n_flowgates: int = 10,
) -> tuple[pd.DataFrame, str]:
    """
    Run generate_narrative() for every hour of target_date for the top-N flowgates
    (by predicted probability) where probability > optimal_threshold.

    Returns (DataFrame[datetime, flowgate_id, probability, narrative], market_summary_str).
    Saves to data/results/daily_explanations/{date}.csv with summary as first comment row.
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()

    day_start = pd.Timestamp(target_date, tz="UTC")
    day_end   = day_start + pd.Timedelta(hours=23)

    # Load eval data for thresholds and status
    eval_df  = pd.read_csv(EVAL_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    prod_ids = train_df[train_df["model_status"] == "production"]["flowgate_id"].tolist()

    # Build threshold lookup from eval_df
    thresh_map = eval_df.set_index("flowgate_id")["optimal_threshold"].to_dict()

    hours = pd.date_range(day_start, day_end, freq="h", tz="UTC")

    rows: list[dict] = []

    for fg_id in prod_ids:
        threshold = thresh_map.get(fg_id, 0.5)
        try:
            base_model, calibrator, uses_calibration = _load_model(fg_id)
        except FileNotFoundError:
            continue
        features = _get_features(fg_id, base_model)

        try:
            df_full = pd.read_parquet(_parquet_path(fg_id))
            needs_seasonal = (
                "binding_rate_same_month_prior_year" in features
                or "days_since_last_binding_rolling_14d" in features
            )
            if needs_seasonal:
                df_full = add_seasonal_features(df_full)
        except Exception:
            continue

        valid_hours = [h for h in hours if h in df_full.index]
        if not valid_hours:
            continue

        X_day = df_full.loc[valid_hours, features]
        raw_probs = safe_predict_proba(base_model, X_day)

        if calibrator is not None:
            probs = np.clip(calibrator.predict(raw_probs.reshape(-1, 1)).ravel(), 0.001, 0.999)
        else:
            probs = raw_probs

        for hour, prob in zip(valid_hours, probs):
            if prob > threshold:
                rows.append({
                    "datetime":    hour,
                    "flowgate_id": fg_id,
                    "probability": float(prob),
                    "_threshold":  threshold,
                })

    if not rows:
        empty = pd.DataFrame(columns=["datetime", "flowgate_id", "probability", "narrative"])
        return empty, f"{target_date.strftime('%B')} {target_date.day} congestion summary: no constraints flagged above threshold."

    candidates = (
        pd.DataFrame(rows)
        .sort_values("probability", ascending=False)
        .head(top_n_flowgates * len(hours))
    )

    # Per-hour: keep top_n_flowgates (sort then take head per group)
    per_hour = (
        candidates
        .sort_values(["datetime", "probability"], ascending=[True, False])
        .groupby("datetime", sort=False)
        .head(top_n_flowgates)
        .reset_index(drop=True)
    )

    # Generate narratives (per_hour already sorted by [datetime, probability desc])
    narratives: list[dict] = []
    for _, r in per_hour.iterrows():
        try:
            narr = generate_narrative(r["flowgate_id"], r["datetime"])
        except Exception as exc:
            narr = f"[explanation unavailable: {exc}]"
        narratives.append({
            "datetime":    r["datetime"],
            "flowgate_id": r["flowgate_id"],
            "probability": r["probability"],
            "narrative":   narr,
        })

    # Collapse runs of 3+ consecutive hours with identical probability per flowgate
    narratives = _dedup_consecutive_runs(narratives)

    result = pd.DataFrame(narratives)

    # Generate market summary from final deduplicated narratives
    summary = generate_market_summary(result, target_date)

    # Write CSV: summary as a leading comment row, then the data
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DAILY_DIR / f"{target_date}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {summary}\n")
        result.to_csv(fh, index=False, lineterminator="\n")

    return result, summary


# ── CLI entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 1 SHAP explainability")
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Run explain_day for this date (YYYY-MM-DD). Default: run global importance.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N flowgates per hour for explain_day (default 10).",
    )
    args = parser.parse_args()

    if args.day:
        print(f"Running explain_day({args.day}, top_n_flowgates={args.top_n}) ...")
        result, summary = explain_day(args.day, top_n_flowgates=args.top_n)
        out_path = DAILY_DIR / f"{args.day}.csv"
        print(f"Saved {len(result)} rows to {out_path}")
        print()
        print("Market summary:")
        print(summary)
        print()
        print("First 5 narratives:")
        print("-" * 70)
        for _, row in result.head(5).iterrows():
            dt = row["datetime"]
            ts_str = dt if isinstance(dt, str) else pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M UTC")
            print(f"[{ts_str}] {row['flowgate_id']}  p={row['probability']:.3f}")
            print(row["narrative"])
            print()
    else:
        print("Part 1: Computing global SHAP importance for all production models ...")
        run_global_importance_all(verbose=True)


if __name__ == "__main__":
    main()
