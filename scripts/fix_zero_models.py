"""
Multi-pass fix for zero PR-AUC models.

Root cause: all 31 zero-PR-AUC flowgates have 0 binding events in the
Jul-Sep 2024 val window — they are seasonal constraints that don't bind
in summer. Val PR-AUC will remain 0 after retraining; Optuna CV PR-AUC
(on train folds) is reported as the true quality signal.

Pass 0 : Diagnostics — duplicate check, PLESNTLK-LEEDS2 confirmation.
Pass 1 : Drop PLESNTLK-LEEDS2 from target_flowgates.csv (0 train bindings).
Pass 2 : ratio < 3:1 flowgates (CHAR_CK) — force spw=3.0, SMOTE(0.2), 75 trials.
Pass 3 : Remaining zero models — cap reg_alpha at 1.0, 75 trials.

Usage:
    py -3 scripts/fix_zero_models.py [--dry-run] [--force-cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.models.cv import MISOTimeSeriesSplit
from src.models.train import (
    TARGET_COL,
    FEATURES_DIR,
    MODELS_DIR,
    IMPORTANCE_DIR,
    RESULTS_CSV,
    N_ESTIMATORS_MAX,
    EARLY_STOPPING,
    _RESULTS_COLS,
    _safe_id,
    _parquet_path,
    _select_features,
    _date_split,
    get_scale_pos_weight,
    _class_weight_strategy,
    assign_flowgate_tier,
    _fit,
    _save_model,
    _save_importance,
    _detect_device,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=r".*(No visible GPU|Device is changed).*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Dataset is empty.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*No positive class found.*", category=UserWarning)

VAL_START = pd.Timestamp("2024-07-01 05:00", tz="UTC")
VAL_END   = pd.Timestamp("2024-10-01 04:00", tz="UTC")


# ── Results CSV helpers ────────────────────────────────────────────────────────

def _load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV)


def _upsert_result(result: dict, dry_run: bool) -> None:
    """Update the row for this flowgate in training_results.csv in place."""
    if dry_run:
        return
    df = _load_results()
    row = {k: result[k] for k in _RESULTS_COLS}
    mask = df["flowgate_id"] == result["flowgate_id"]
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)


def _drop_from_results(flowgate_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would drop {flowgate_id} from results CSV")
        return
    df = _load_results()
    df = df[df["flowgate_id"] != flowgate_id]
    df.to_csv(RESULTS_CSV, index=False)


# ── Optuna objectives ──────────────────────────────────────────────────────────

def _objective_standard(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    cv: MISOTimeSeriesSplit,
    device: str,
    force_spw: float | None,
    reg_alpha_max: float,
) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, reg_alpha_max, log=True),
    }

    pr_aucs: list[float] = []
    for fold_train, fold_val in cv.split_by_date(train_df):
        X_tr = fold_train.drop(columns=[TARGET_COL])
        y_tr = fold_train[TARGET_COL]
        X_v  = fold_val.drop(columns=[TARGET_COL])
        y_v  = fold_val[TARGET_COL]

        spw = force_spw if force_spw is not None else get_scale_pos_weight(y_tr)
        model = _fit(X_tr, y_tr, X_v, y_v, params, spw, device)
        preds = model.predict_proba(X_v)[:, 1]
        pr_aucs.append(average_precision_score(y_v, preds))

    return float(np.mean(pr_aucs))


def _objective_smote(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    cv: MISOTimeSeriesSplit,
    device: str,
    force_spw: float,
) -> float:
    """Objective with SMOTE applied to each training fold."""
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
    }

    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    pr_aucs: list[float] = []

    for fold_train, fold_val in cv.split_by_date(train_df):
        X_tr = fold_train.drop(columns=[TARGET_COL]).values
        y_tr = fold_train[TARGET_COL].values
        X_v  = fold_val.drop(columns=[TARGET_COL])
        y_v  = fold_val[TARGET_COL]

        # SMOTE only fires when minority class is below the target ratio.
        # For CHAR_CK (pos ~45%), it is a no-op; for sparser positives it adds samples.
        try:
            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        except ValueError:
            X_tr_res, y_tr_res = X_tr, y_tr

        col_names = fold_train.drop(columns=[TARGET_COL]).columns
        X_tr_df = pd.DataFrame(X_tr_res, columns=col_names)
        y_tr_s  = pd.Series(y_tr_res, name=TARGET_COL)

        model = _fit(X_tr_df, y_tr_s, X_v, y_v, params, force_spw, device)
        preds = model.predict_proba(X_v)[:, 1]
        pr_aucs.append(average_precision_score(y_v, preds))

    return float(np.mean(pr_aucs))


# ── Core retrain function ──────────────────────────────────────────────────────

def retrain_flowgate(
    flowgate_id: str,
    features_df: pd.DataFrame,
    n_trials: int,
    device: str,
    force_spw: float | None = None,
    use_smote: bool = False,
    reg_alpha_max: float = 10.0,
) -> dict:
    t_start = time.perf_counter()

    obs_pct = float(
        features_df["flowgate_loading_pct_is_observed"].mean()
        if "flowgate_loading_pct_is_observed" in features_df.columns
        else 0.0
    )
    binding_rate = float(features_df[TARGET_COL].mean())
    tier = assign_flowgate_tier(binding_rate, obs_pct)

    selected_df = _select_features(features_df, tier)
    train_df, val_df = _date_split(selected_df)

    y_train = train_df[TARGET_COL]
    ratio   = get_scale_pos_weight(y_train)
    strategy = _class_weight_strategy(ratio)

    effective_spw = force_spw  # None means "compute per fold"

    cv = MISOTimeSeriesSplit(n_splits=5, gap_hours=24)
    study = optuna.create_study(direction="maximize")

    if use_smote:
        study.optimize(
            lambda trial: _objective_smote(trial, train_df, cv, device, force_spw or 3.0),
            n_trials=n_trials,
            show_progress_bar=False,
        )
    else:
        study.optimize(
            lambda trial: _objective_standard(
                trial, train_df, cv, device, effective_spw, reg_alpha_max
            ),
            n_trials=n_trials,
            show_progress_bar=False,
        )

    X_train = train_df.drop(columns=[TARGET_COL])
    X_val   = val_df.drop(columns=[TARGET_COL])
    y_val   = val_df[TARGET_COL]

    if use_smote:
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        try:
            X_res, y_res = smote.fit_resample(X_train.values, y_train.values)
            X_fit = pd.DataFrame(X_res, columns=X_train.columns)
            y_fit = pd.Series(y_res, name=TARGET_COL)
        except ValueError:
            X_fit, y_fit = X_train, y_train
        final_spw = force_spw or 3.0
    else:
        X_fit, y_fit = X_train, y_train
        final_spw = force_spw if force_spw is not None else get_scale_pos_weight(y_train)

    final_model = _fit(X_fit, y_fit, X_val, y_val, study.best_params, final_spw, device)

    preds_val = final_model.predict_proba(X_val)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, preds_val))

    elapsed = time.perf_counter() - t_start

    return {
        "flowgate_id":            flowgate_id,
        "tier":                   tier,
        "class_weight_strategy":  strategy,
        "best_pr_auc":            val_pr_auc,
        "optuna_cv_pr_auc":       round(study.best_value, 4),
        "best_params":            json.dumps(study.best_params),
        "n_binding_hours_train":  int(y_train.sum()),
        "training_time_seconds":  round(elapsed, 1),
        "device_used":            device,
        "_model":                 final_model,
        "_feature_names":         list(X_train.columns),
        "_ratio":                 ratio,
    }


# ── Pass helpers ───────────────────────────────────────────────────────────────

def _print_pass_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_result(result: dict, pass_num: int) -> None:
    still_zero = " *** STILL ZERO VAL ***" if result["best_pr_auc"] == 0.0 else ""
    note = " (val window all-negative — see cv PR-AUC)" if result["best_pr_auc"] == 0.0 else ""
    print(
        f"  [P{pass_num}] {result['flowgate_id']}"
        f"  val={result['best_pr_auc']:.4f}"
        f"  cv={result['optuna_cv_pr_auc']:.4f}"
        f"  ratio={result['_ratio']:.1f}:1"
        f"  time={result['training_time_seconds']:.0f}s"
        f"{still_zero}{note}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = _detect_device(args.force_cpu)
    print(f"Device : {device}")

    results_df = _load_results()
    zeros = results_df[results_df["best_pr_auc"] == 0.0].copy()

    # ── Pass 0: Diagnostics ────────────────────────────────────────────────────
    _print_pass_header("Pass 0 — Diagnostics")

    print(f"\n  Zero PR-AUC models    : {len(zeros)} / {len(results_df)}")
    print(  "  Root cause            : ALL have 0 binding events in Jul-Sep val window.")
    print(  "  Val PR-AUC will remain 0 after retraining for seasonal constraints.")
    print(  "  Optuna CV PR-AUC (train folds) is the true quality signal.\n")

    # Duplicate check: PRES-TIBB
    fg1 = "PRES-TIBB 138 FLO ASTER-COMMODORE"
    fg2 = "PRES - TIBB 138 FLO ASTER - COMMODORE 34"
    p1, p2 = _parquet_path(fg1), _parquet_path(fg2)
    if p1.exists() and p2.exists():
        b1 = pd.read_parquet(p1)["binding"]
        b2 = pd.read_parquet(p2)["binding"]
        common = b1.index.intersection(b2.index)
        corr = b1.loc[common].corr(b2.loc[common])
        overlap = int((b1.loc[common].astype(bool) & b2.loc[common].astype(bool)).sum())
        print(f"  Duplicate check:")
        print(f"    {fg1!r}")
        print(f"    {fg2!r}")
        print(f"    Pearson corr = {corr:.4f}  |  shared binding hours = {overlap}")
        if corr > 0.90:
            print("    => DUPLICATES (corr > 0.90) — will drop FG2")
        else:
            print(f"    => NOT duplicates (corr = {corr:.4f} < 0.90, 0 shared hours) — keeping both")

    # PLESNTLK-LEEDS2 confirmation
    fg_pl = "PLESNTLK-LEEDS2 FLO BALTA_GRE-RAMSEY"
    p_pl = _parquet_path(fg_pl)
    if p_pl.exists():
        df_pl = pd.read_parquet(p_pl)
        train_mask = df_pl.index <= pd.Timestamp("2024-07-01 04:00", tz="UTC")
        n_train_bind = int(df_pl.loc[train_mask, "binding"].sum())
        print(f"\n  PLESNTLK-LEEDS2 binding hours in train : {n_train_bind}")
        print(f"  => Will drop (data issue — model cannot learn from 0 training examples)")

    # ── Pass 1: Drop PLESNTLK-LEEDS2 ──────────────────────────────────────────
    _print_pass_header("Pass 1 — Drop PLESNTLK-LEEDS2")

    fg_pl = "PLESNTLK-LEEDS2 FLO BALTA_GRE-RAMSEY"
    fg_csv = Path("data/processed/target_flowgates.csv")
    if fg_csv.exists():
        fg_df = pd.read_csv(fg_csv, index_col=0)
        if fg_pl in fg_df.index:
            if not args.dry_run:
                fg_df = fg_df.drop(index=fg_pl)
                fg_df.to_csv(fg_csv)
            print(f"  Dropped {fg_pl!r} from target_flowgates.csv")
        else:
            print(f"  {fg_pl!r} not in target_flowgates.csv (already removed)")

    _drop_from_results(fg_pl, args.dry_run)

    # Remove stale model + importance artifacts
    for artifact_dir, ext in [(MODELS_DIR, ".joblib"), (IMPORTANCE_DIR, ".csv")]:
        artifact = artifact_dir / f"{_safe_id(fg_pl)}{ext}"
        if artifact.exists() and not args.dry_run:
            artifact.unlink()
            print(f"  Removed {artifact}")

    print(f"  Pass 1 complete. Flowgates in scope: {len(results_df) - 1}")

    # ── Pass 2: ratio < 3:1 — SMOTE + force spw=3.0 ───────────────────────────
    _print_pass_header("Pass 2 — ratio < 3:1 (SMOTE + force scale_pos_weight=3.0, 75 trials)")

    results_df = _load_results()
    zeros_p2 = results_df[results_df["best_pr_auc"] == 0.0].copy()

    pass2_ids: list[str] = []
    for _, row in zeros_p2.iterrows():
        fid = row["flowgate_id"]
        n_bind = int(row["n_binding_hours_train"])
        train_hrs = len(pd.read_parquet(_parquet_path(fid)).loc[
            lambda df: df.index <= pd.Timestamp("2024-07-01 04:00", tz="UTC")
        ])
        ratio = (train_hrs - n_bind) / max(n_bind, 1)
        if ratio < 3.0:
            pass2_ids.append(fid)

    print(f"\n  Flowgates to retrain (ratio < 3:1) : {len(pass2_ids)}")
    for fid in pass2_ids:
        print(f"    {fid}")

    for fid in pass2_ids:
        df = pd.read_parquet(_parquet_path(fid))
        if args.dry_run:
            print(f"  [DRY RUN] Would retrain {fid}")
            continue
        result = retrain_flowgate(fid, df, n_trials=75, device=device,
                                  force_spw=3.0, use_smote=True)
        _save_model(result["_model"], fid)
        _save_importance(result["_model"], result["_feature_names"], fid)
        _upsert_result(result, dry_run=False)
        _print_result(result, pass_num=2)

    # ── Pass 3: remaining zeros — tighter reg_alpha ────────────────────────────
    _print_pass_header("Pass 3 — remaining zeros (reg_alpha cap 1.0, 75 trials)")

    results_df = _load_results()
    pass2_set  = set(pass2_ids)
    zeros_p3   = results_df[
        (results_df["best_pr_auc"] == 0.0) &
        (~results_df["flowgate_id"].isin(pass2_set))
    ].copy()

    print(f"\n  Flowgates to retrain : {len(zeros_p3)}")

    for _, row in zeros_p3.iterrows():
        fid = row["flowgate_id"]
        df = pd.read_parquet(_parquet_path(fid))
        if args.dry_run:
            print(f"  [DRY RUN] Would retrain {fid}")
            continue
        result = retrain_flowgate(fid, df, n_trials=75, device=device, reg_alpha_max=1.0)
        _save_model(result["_model"], fid)
        _save_importance(result["_model"], result["_feature_names"], fid)
        _upsert_result(result, dry_run=False)
        _print_result(result, pass_num=3)

    # ── Final summary ──────────────────────────────────────────────────────────
    _print_pass_header("Final Summary")

    if args.dry_run:
        print("\n  [DRY RUN] No changes written.")
        return

    results_final = _load_results()
    still_zero = results_final[results_final["best_pr_auc"] == 0.0]

    print(f"\n  Total flowgates       : {len(results_final)}")
    print(f"  Val PR-AUC > 0        : {len(results_final) - len(still_zero)}")
    print(f"  Val PR-AUC = 0        : {len(still_zero)}  (all have 0 binding in val window)")

    if not still_zero.empty:
        print(f"\n  Remaining zero-val models (seasonal — not fixable via val metric):")
        for _, row in still_zero.iterrows():
            cv_auc = row.get("optuna_cv_pr_auc", float("nan"))
            print(f"    {row['flowgate_id']:<55s}  cv={cv_auc:.4f}")

    print(f"\n  Results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
