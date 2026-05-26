"""
Retrain zero-val flowgates (CV >= MIN_CV) with two new seasonal features.

Features added on-the-fly (NOT written back to parquet):
  binding_rate_same_month_prior_year  — prior-year same-month avg binding rate
  days_since_last_binding_rolling_14d — days since last binding in trailing 14 d

Both features are backward-looking only: no label leakage.

For flowgates where val window has 0 binding events, val PR-AUC stays 0 regardless
of features. The gain shows in optuna_cv_pr_auc and future test-set performance.

Usage:
    py -3 scripts/retrain_seasonal_features.py
    py -3 scripts/retrain_seasonal_features.py --min-cv 0.50
    py -3 scripts/retrain_seasonal_features.py --n-trials 100
    py -3 scripts/retrain_seasonal_features.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.models.train import (
    TARGET_COL,
    FEATURES_DIR,
    MODELS_DIR,
    IMPORTANCE_DIR,
    RESULTS_CSV,
    N_ESTIMATORS_MAX,
    EARLY_STOPPING,
    _RESULTS_COLS,
    _TRAIN_END,
    _VAL_START,
    _VAL_END,
    _safe_id,
    _parquet_path,
    _select_features,
    _date_split,
    get_scale_pos_weight,
    _class_weight_strategy,
    _effective_scale_pos_weight,
    assign_flowgate_tier,
    _fit,
    _save_model,
    _save_importance,
    _detect_device,
)
from src.models.cv import MISOTimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

MIN_CV = 0.40   # "moderate" CV threshold


# ── New features ────────────────────────────────────────────────────────────────

def _binding_rate_prior_year(binding: pd.Series) -> pd.Series:
    """
    For each hourly row, look up the average binding rate for the same
    calendar month in the prior year. NaN for rows where prior-year data
    is unavailable (e.g., all 2023 rows when training starts in 2023).
    """
    monthly = binding.groupby([binding.index.year, binding.index.month]).mean()
    # Dict for O(1) lookup: {(year, month): rate}
    rate_map: dict[tuple[int, int], float] = {k: float(v) for k, v in monthly.items()}

    years  = binding.index.year - 1   # prior year
    months = binding.index.month

    values = [rate_map.get((int(y), int(m)), np.nan) for y, m in zip(years, months)]
    return pd.Series(values, index=binding.index,
                     name="binding_rate_same_month_prior_year", dtype="float32")


def _days_since_last_binding(binding: pd.Series, window_days: int = 14) -> pd.Series:
    """
    Days since the most recent binding event in the trailing window,
    EXCLUDING the current hour (backward-looking only).
    Returns window_days when no binding occurred in the window.
    """
    window_hours = window_days * 24
    vals   = binding.values
    result = np.full(len(vals), float(window_days), dtype="float32")
    last_bind = -1  # index of last seen binding; -1 = none yet

    for i in range(len(vals)):
        # Compute feature for row i using history [0 .. i-1]
        if last_bind >= 0:
            hours_ago = i - last_bind
            if hours_ago <= window_hours:
                result[i] = hours_ago / 24.0
            # else: stays at window_days (no binding in window)
        # Update tracker after using it — never leaks current row
        if vals[i] == 1:
            last_bind = i

    return pd.Series(result, index=binding.index,
                     name="days_since_last_binding_rolling_14d")


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach both seasonal features. Returns new DataFrame."""
    df = df.copy()
    df["binding_rate_same_month_prior_year"]  = _binding_rate_prior_year(df[TARGET_COL])
    df["days_since_last_binding_rolling_14d"] = _days_since_last_binding(df[TARGET_COL])
    return df


# ── CSV upsert ──────────────────────────────────────────────────────────────────

def _upsert_result(result: dict, dry_run: bool) -> None:
    row = {k: result[k] for k in _RESULTS_COLS}
    if dry_run:
        print(f"    [dry-run] would upsert {result['flowgate_id']}")
        return
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        mask = df["flowgate_id"] == result["flowgate_id"]
        if mask.any():
            for col, val in row.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False)


# ── Optuna objective with new features ─────────────────────────────────────────

def _objective(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    cv: MISOTimeSeriesSplit,
    strategy: str,
    device: str,
) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
    }
    pr_aucs: list[float] = []
    for fold_train, fold_val in cv.split_by_date(train_df):
        X_tr = fold_train.drop(columns=[TARGET_COL])
        y_tr = fold_train[TARGET_COL]
        X_v  = fold_val.drop(columns=[TARGET_COL])
        y_v  = fold_val[TARGET_COL]
        spw   = _effective_scale_pos_weight(y_tr, strategy)
        model = _fit(X_tr, y_tr, X_v, y_v, params, spw, device)
        preds = model.predict_proba(X_v)[:, 1]
        pr_aucs.append(average_precision_score(y_v, preds))
    return float(np.mean(pr_aucs))


# ── Per-flowgate retrain ────────────────────────────────────────────────────────

def retrain_flowgate(
    flowgate_id: str,
    features_df: pd.DataFrame,
    n_trials: int,
    device: str,
) -> dict:
    t_start = time.perf_counter()

    # Add seasonal features before any split
    features_df = add_seasonal_features(features_df)

    # Tier / feature selection / split (same logic as train.py)
    obs_pct = float(
        features_df["flowgate_loading_pct_is_observed"].mean()
        if "flowgate_loading_pct_is_observed" in features_df.columns
        else 0.0
    )
    binding_rate = float(features_df[TARGET_COL].mean())
    tier = assign_flowgate_tier(binding_rate, obs_pct)

    selected_df = _select_features(features_df, tier)
    train_df, val_df = _date_split(selected_df)

    y_train  = train_df[TARGET_COL]
    ratio    = get_scale_pos_weight(y_train)
    strategy = _class_weight_strategy(ratio)

    cv = MISOTimeSeriesSplit(n_splits=5, gap_hours=24)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, train_df, cv, strategy, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    X_train   = train_df.drop(columns=[TARGET_COL])
    X_val     = val_df.drop(columns=[TARGET_COL])
    y_val     = val_df[TARGET_COL]
    final_spw = _effective_scale_pos_weight(y_train, strategy)

    final_model = _fit(
        X_train, y_train,
        X_val,   y_val,
        study.best_params,
        final_spw,
        device,
    )

    preds_val   = final_model.predict_proba(X_val)[:, 1]
    best_pr_auc = float(average_precision_score(y_val, preds_val))
    elapsed     = time.perf_counter() - t_start

    return {
        "flowgate_id":           flowgate_id,
        "tier":                  tier,
        "class_weight_strategy": strategy,
        "best_pr_auc":           best_pr_auc,
        "optuna_cv_pr_auc":      round(study.best_value, 4),
        "best_params":           json.dumps(study.best_params),
        "n_binding_hours_train": int(y_train.sum()),
        "training_time_seconds": round(elapsed, 1),
        "device_used":           device,
        "_model":                final_model,
        "_feature_names":        list(X_train.columns),
        "_ratio":                ratio,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cv",   type=float, default=MIN_CV)
    parser.add_argument("--n-trials", type=int,   default=75)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    device = _detect_device(force_cpu=False)
    print(f"Device : {device}\n")

    if not RESULTS_CSV.exists():
        print("ERROR: data/results/training_results.csv not found.")
        sys.exit(1)

    results_df  = pd.read_csv(RESULTS_CSV)
    zero_val    = results_df[results_df["best_pr_auc"] == 0.0]
    moderate_cv = zero_val[zero_val["optuna_cv_pr_auc"] >= args.min_cv]

    fg_list = moderate_cv.sort_values("optuna_cv_pr_auc", ascending=False)["flowgate_id"].tolist()

    print("=" * 70)
    print(f"  Retrain: zero-val AND CV >= {args.min_cv:.2f}")
    print(f"  Flowgates selected : {len(fg_list)}")
    print(f"  New features: binding_rate_same_month_prior_year,")
    print(f"                days_since_last_binding_rolling_14d")
    print(f"  n_trials={args.n_trials}  dry_run={args.dry_run}")
    print("=" * 70)
    for fid in fg_list:
        row = results_df[results_df["flowgate_id"] == fid].iloc[0]
        print(f"    {fid:<55}  prev_cv={row['optuna_cv_pr_auc']:.4f}")
    print()

    n_done  = 0
    n_error = 0

    for i, fg_id in enumerate(fg_list, start=1):
        parquet = _parquet_path(fg_id)
        if not parquet.exists():
            print(f"[{i:>2}/{len(fg_list)}] {fg_id}  -- SKIP (parquet missing)")
            continue

        features_df = pd.read_parquet(parquet)

        print(f"[{i:>2}/{len(fg_list)}] {fg_id}", flush=True)
        try:
            result = retrain_flowgate(fg_id, features_df, args.n_trials, device)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_error += 1
            continue

        prev_row = results_df[results_df["flowgate_id"] == fg_id].iloc[0]
        cv_delta = result["optuna_cv_pr_auc"] - float(prev_row["optuna_cv_pr_auc"])
        direction = "+" if cv_delta > 0 else ("-" if cv_delta < 0 else "=")

        print(
            f"  val={result['best_pr_auc']:.4f}  "
            f"cv={result['optuna_cv_pr_auc']:.4f} ({direction}{abs(cv_delta):.4f})  "
            f"time={result['training_time_seconds']:.0f}s"
        )

        if not args.dry_run:
            _save_model(result["_model"], fg_id)
            _save_importance(result["_model"], result["_feature_names"], fg_id)
            _upsert_result(result, dry_run=False)

        n_done += 1

    print()
    print("=" * 70)
    print(f"  Done. Retrained: {n_done}  Errors: {n_error}")
    print(f"  Results -> {RESULTS_CSV}")
    print("=" * 70)
    print()
    print("NOTE: These models expect two new features at inference time:")
    print("  binding_rate_same_month_prior_year")
    print("  days_since_last_binding_rolling_14d")
    print("Use scripts/retrain_seasonal_features.add_seasonal_features(df)")
    print("before passing data to model.predict_proba().")


if __name__ == "__main__":
    main()
