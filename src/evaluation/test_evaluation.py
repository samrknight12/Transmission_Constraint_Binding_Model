"""
Test-set evaluation for Layer 1 binding classifiers.

Evaluates all models where model_status IN ("production", "marginal").

Splits (UTC; MISO EST = UTC-5, fixed, no DST):
    Val  : 2024-07-01 05:00 UTC - 2024-10-01 04:00 UTC
    Test : 2024-10-01 05:00 UTC - 2025-01-01 04:00 UTC  (held-out, never fit here)

Threshold is derived from the validation set only.
Isotonic calibrator (if needed) is fit on the validation set only.
Nothing is fit to test data.

Usage:
    py -3 src/evaluation/test_evaluation.py
    py -3 src/evaluation/test_evaluation.py --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.train import (
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COL,
    _safe_id,
    _VAL_START,
    _VAL_END,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAINING_CSV = Path("data/results/training_results.csv")
EVAL_CSV     = RESULTS_DIR / "test_evaluation.csv"

# ── Test window ────────────────────────────────────────────────────────────────
_TEST_START = pd.Timestamp("2024-10-01 05:00", tz="UTC")   # 2024-10-01 00:00 EST
_TEST_END   = pd.Timestamp("2025-01-01 04:00", tz="UTC")   # 2024-12-31 23:00 EST

# ── Calibration ────────────────────────────────────────────────────────────────
ECE_THRESHOLD = 0.05   # flag production models above this for isotonic recalibration

EVAL_COLS = [
    "flowgate_id",
    "model_status",
    "test_pr_auc",
    "test_brier_score",
    "optimal_threshold",
    "test_precision_at_threshold",
    "test_recall_at_threshold",
    "test_f1_at_threshold",
    "n_binding_hours_test",
    "n_total_hours_test",
    "ece_raw",
    "ece_calibrated",
    "calibration_status",
]


# ── Threshold from val set ─────────────────────────────────────────────────────

def _optimal_threshold_from_val(y_val: pd.Series, probs_val: np.ndarray) -> float:
    """
    Scan thresholds 0.01..0.99 and return the one maximising F1 on the val set.
    Falls back to 0.5 if val has no positives or no threshold yields TP > 0.
    """
    if y_val.sum() == 0:
        return 0.5

    best_f1     = -1.0
    best_thresh = 0.5

    for t in np.arange(0.01, 1.00, 0.01):
        preds = (probs_val >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = round(float(t), 2)

    return best_thresh


# ── Calibration helpers ────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Uses calibration_curve to obtain per-bin accuracy and confidence, then
    weights by bin occupancy: ECE = sum(|B_m|/n * |acc(B_m) - conf(B_m)|).
    """
    if y_true.sum() == 0:
        return float("nan")

    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    n        = len(y_true)
    ece      = 0.0

    # calibration_curve gives per-bin fraction_of_positives and mean_predicted_value
    # We bin manually so we can also get occupancy counts for weighting.
    for lo, hi in zip(bins[:-1], bins[1:]):
        # last bin is closed on the right
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (cnt / n) * abs(acc - conf)

    return float(ece)


def _fit_isotonic(
    y_val: np.ndarray, probs_val: np.ndarray
) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs_val, y_val)
    return cal


# ── Per-flowgate evaluation ────────────────────────────────────────────────────

def _load_features(flowgate_id: str) -> pd.DataFrame | None:
    path = FEATURES_DIR / f"{_safe_id(flowgate_id)}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def evaluate_flowgate(
    flowgate_id: str,
    model_status: str,
    model: object,
) -> dict | None:
    df = _load_features(flowgate_id)
    if df is None:
        return None

    feature_cols = list(model.feature_names_in_)

    val_df  = df[(df.index >= _VAL_START) & (df.index <= _VAL_END)].copy()
    test_df = df[(df.index >= _TEST_START) & (df.index <= _TEST_END)].copy()

    if len(test_df) == 0:
        return None

    y_val  = val_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    if len(set(feature_cols) - set(test_df.columns)) > 0:
        missing = set(feature_cols) - set(test_df.columns)
        print(f"  WARN {flowgate_id}: {len(missing)} features missing, skipping.")
        return None

    X_val  = val_df[feature_cols] if len(val_df) > 0 else pd.DataFrame(columns=feature_cols)
    X_test = test_df[feature_cols]

    # Always compute val probs when val rows exist — needed for calibration
    if len(val_df) > 0:
        probs_val = model.predict_proba(X_val)[:, 1]
    else:
        probs_val = None

    threshold = (
        _optimal_threshold_from_val(y_val, probs_val)
        if probs_val is not None and y_val.sum() > 0
        else 0.5
    )

    probs_test = model.predict_proba(X_test)[:, 1]

    n_pos = int(y_test.sum())
    n_tot = int(len(y_test))

    base = {
        "flowgate_id":                 flowgate_id,
        "model_status":                model_status,
        "test_brier_score":            float(brier_score_loss(y_test, probs_test)),
        "optimal_threshold":           threshold,
        "n_binding_hours_test":        n_pos,
        "n_total_hours_test":          n_tot,
        "_probs_test":                 probs_test,
        "_probs_val":                  probs_val,
        "_y_test":                     y_test,
        "_y_val":                      y_val,
    }

    if n_pos == 0:
        return {**base,
                "test_pr_auc":                 0.0,
                "test_precision_at_threshold": 0.0,
                "test_recall_at_threshold":    0.0,
                "test_f1_at_threshold":        0.0}

    preds_test = (probs_test >= threshold).astype(int)
    return {**base,
            "test_pr_auc":                 float(average_precision_score(y_test, probs_test)),
            "test_precision_at_threshold": float(precision_score(y_test, preds_test, zero_division=0)),
            "test_recall_at_threshold":    float(recall_score(y_test, preds_test, zero_division=0)),
            "test_f1_at_threshold":        float(f1_score(y_test, preds_test, zero_division=0))}


# ── Top-K hourly precision ─────────────────────────────────────────────────────

def _top_k_precision(
    prob_rows: dict[str, pd.Series],
    bind_rows: dict[str, pd.Series],
    k: int,
) -> float:
    """
    For each hour in the test window, rank all flowgates by predicted probability,
    take the top-k, and compute what fraction actually bound.
    Returns mean precision across all hours that have >= k flowgates with predictions.
    """
    prob_df = pd.DataFrame(prob_rows)
    bind_df = pd.DataFrame(bind_rows)

    common_idx = prob_df.index.intersection(bind_df.index)
    prob_df    = prob_df.loc[common_idx]
    bind_df    = bind_df.loc[common_idx]

    precisions: list[float] = []
    for dt in prob_df.index:
        row_probs = prob_df.loc[dt].dropna()
        if len(row_probs) < k:
            continue
        top_k_fg     = row_probs.nlargest(k).index
        actual_top_k = bind_df.loc[dt, top_k_fg]
        if actual_top_k.isna().all():
            continue
        precisions.append(float(actual_top_k.mean()))

    return float(np.mean(precisions)) if precisions else float("nan")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(TRAINING_CSV)
    eligible = training[training["model_status"].isin(["production", "marginal"])].copy()
    print(f"Eligible models: {len(eligible)}  "
          f"(production={(eligible['model_status']=='production').sum()}, "
          f"marginal={(eligible['model_status']=='marginal').sum()})")
    print(f"Test window: {_TEST_START.date()} to {_TEST_END.date()}\n")

    results: list[dict]            = []
    prob_series: dict[str, pd.Series] = {}
    bind_series: dict[str, pd.Series] = {}
    n_skip = 0

    # ── Pass 1: per-flowgate inference ────────────────────────────────────────
    for i, (_, row) in enumerate(eligible.iterrows(), start=1):
        fg_id        = row["flowgate_id"]
        model_status = row["model_status"]
        model_path   = MODELS_DIR / f"{_safe_id(fg_id)}.joblib"

        if not model_path.exists():
            print(f"[{i:>3}] {fg_id}  -- SKIP (model file missing)")
            n_skip += 1
            continue

        model   = joblib.load(model_path)
        metrics = evaluate_flowgate(fg_id, model_status, model)

        if metrics is None:
            print(f"[{i:>3}] {fg_id}  -- SKIP (no test data)")
            n_skip += 1
            continue

        df_raw     = pd.read_parquet(FEATURES_DIR / f"{_safe_id(fg_id)}.parquet")
        df_raw.index = pd.to_datetime(df_raw.index, utc=True)
        test_slice = df_raw[(df_raw.index >= _TEST_START) & (df_raw.index <= _TEST_END)]

        prob_series[fg_id] = pd.Series(metrics["_probs_test"], index=test_slice.index, name=fg_id)
        bind_series[fg_id] = pd.Series(metrics["_y_test"].values, index=test_slice.index, name=fg_id)

        flag = " *no test binding*" if metrics["n_binding_hours_test"] == 0 else ""
        print(
            f"[{i:>3}] {fg_id}"
            f"  pr_auc={metrics['test_pr_auc']:.4f}"
            f"  brier={metrics['test_brier_score']:.4f}"
            f"  thresh={metrics['optimal_threshold']:.2f}"
            f"  f1={metrics['test_f1_at_threshold']:.4f}"
            f"  bind={metrics['n_binding_hours_test']}/{metrics['n_total_hours_test']}"
            f"{flag}"
        )
        results.append(metrics)

    if not results:
        print("No results to report.")
        return

    # ── Pass 2: calibration (production models with test binding only) ─────────
    print(f"\nCalibration check (production, ECE threshold={ECE_THRESHOLD})...")
    n_needs_cal = 0
    n_cal_ok    = 0

    for r in results:
        # marginal models and zero-binding production models: skip calibration
        if r["model_status"] != "production" or r["n_binding_hours_test"] == 0:
            r["ece_raw"]           = np.nan
            r["ece_calibrated"]    = np.nan
            r["calibration_status"] = "n/a"
            continue

        y_test_arr     = r["_y_test"].values.astype(float)
        probs_test_arr = r["_probs_test"]

        ece_raw = _compute_ece(y_test_arr, probs_test_arr)
        r["ece_raw"] = round(ece_raw, 4)

        # Use calibration_curve for diagnostic output (fraction_pos vs mean_pred)
        frac_pos, mean_pred = calibration_curve(
            r["_y_test"], probs_test_arr, n_bins=10, strategy="uniform"
        )

        if ece_raw <= ECE_THRESHOLD or r["_probs_val"] is None or r["_y_val"].sum() == 0:
            r["ece_calibrated"]    = np.nan
            r["calibration_status"] = "ok"
            n_cal_ok += 1
            print(f"  ok               {r['flowgate_id']:<52}  ECE={ece_raw:.4f}")
            continue

        # Needs calibration — fit isotonic on val, apply to test
        n_needs_cal += 1
        y_val_arr     = r["_y_val"].values.astype(float)
        probs_val_arr = r["_probs_val"]

        calibrator  = _fit_isotonic(y_val_arr, probs_val_arr)
        probs_cal   = calibrator.predict(probs_test_arr)
        ece_cal     = _compute_ece(y_test_arr, probs_cal)

        r["ece_calibrated"]    = round(ece_cal, 4)
        r["calibration_status"] = "needs_calibration"

        # Save calibrated model bundle
        fg_id      = r["flowgate_id"]
        base_model = joblib.load(MODELS_DIR / f"{_safe_id(fg_id)}.joblib")
        cal_path   = MODELS_DIR / f"{_safe_id(fg_id)}_calibrated.joblib"
        joblib.dump(
            {"base_model": base_model, "calibrator": calibrator, "type": "isotonic"},
            cal_path,
        )

        delta = ece_cal - ece_raw
        sign  = "+" if delta >= 0 else "-"
        print(
            f"  NEEDS_CAL        {fg_id:<52}  "
            f"ECE {ece_raw:.4f} -> {ece_cal:.4f} ({sign}{abs(delta):.4f})  "
            f"saved: {cal_path.name}"
        )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    eval_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])[EVAL_COLS]
    eval_df.to_csv(EVAL_CSV, index=False)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    prod_rows = eval_df[eval_df["model_status"] == "production"]
    marg_rows = eval_df[eval_df["model_status"] == "marginal"]

    med_prod  = prod_rows["test_pr_auc"].median()
    med_marg  = marg_rows["test_pr_auc"].median()
    med_brier = eval_df["test_brier_score"].median()
    med_ece   = prod_rows["ece_raw"].dropna().median()

    topk_prec = _top_k_precision(prob_series, bind_series, k=args.top_k)

    # ── Save aggregate metrics (consumed by generate_layer1_summary.py) ────────
    agg_path = RESULTS_DIR / "aggregate_metrics.json"
    with open(agg_path, "w") as _f:
        json.dump({
            "top_k":                 args.top_k,
            "top_k_precision":       round(float(topk_prec), 4),
            "median_brier":          round(float(med_brier), 4),
            "median_prod_pr_auc":    round(float(med_prod), 4),
            "median_marg_pr_auc":    round(float(med_marg), 4),
            "median_ece_production": round(float(med_ece), 4),
            "n_production":          int(len(prod_rows)),
            "n_marginal":            int(len(marg_rows)),
            "n_needs_calibration":   int(n_needs_cal),
            "n_evaluated":           int(len(eval_df)),
        }, _f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Test Evaluation Summary")
    print(sep)
    print(f"  Production models  (N={len(prod_rows):>2}):  median test PR-AUC = {med_prod:.4f}")
    print(f"  Marginal models    (N={len(marg_rows):>2}):  median test PR-AUC = {med_marg:.4f}")
    print(f"  Top-{args.top_k} precision:               {topk_prec*100:.1f}%")
    print(f"  Brier score (median):          {med_brier:.4f}")
    print(f"  ECE median (production):       {med_ece:.4f}")
    print(f"  Models needing calibration:    {n_needs_cal}  (ECE > {ECE_THRESHOLD})")
    print(sep)

    if n_needs_cal:
        print(f"\n  Calibrated models saved to models/saved/*_calibrated.joblib")
        needs_cal_df = eval_df[eval_df["calibration_status"] == "needs_calibration"]
        print(f"  {'Flowgate':<52}  ECE_raw  ECE_cal")
        for _, r in needs_cal_df.sort_values("ece_raw", ascending=False).iterrows():
            print(f"    {r['flowgate_id']:<50}  {r['ece_raw']:.4f}   {r['ece_calibrated']:.4f}")

    no_bind = eval_df[eval_df["n_binding_hours_test"] == 0]
    if len(no_bind):
        print(f"\n  {len(no_bind)} models had 0 binding in test window:")
        for _, r in no_bind.iterrows():
            print(f"    {r['flowgate_id']}  ({r['model_status']})")

    with_bind = eval_df[eval_df["n_binding_hours_test"] > 0].sort_values("test_pr_auc")
    if len(with_bind) >= 5:
        print(f"\n  Bottom 5 by test PR-AUC (have binding):")
        for _, r in with_bind.head(5).iterrows():
            print(f"    {r['flowgate_id']:<50}  pr_auc={r['test_pr_auc']:.4f}")

    print(f"\n  Top 5 by test PR-AUC:")
    for _, r in eval_df.sort_values("test_pr_auc", ascending=False).head(5).iterrows():
        print(f"    {r['flowgate_id']:<50}  pr_auc={r['test_pr_auc']:.4f}")

    print(f"\n  Results -> {EVAL_CSV}")
    print(f"  Skipped: {n_skip}\n")


if __name__ == "__main__":
    main()
