"""
Batch training for MISO Layer 1 binding classifier — all target flowgates.

Splits (fixed EST = UTC-5, no DST):
    Train : 2023-01-01 - 2024-06-30  (Optuna CV + final fit)
    Val   : 2024-07-01 - 2024-09-30  (early stopping + PR-AUC reporting)
    Test  : 2024-10-01 - 2024-12-31  (held out, never touched here)

Usage:
    py -3 src/models/train.py                            # all flowgates
    py -3 src/models/train.py --flowgate CHAR_CK         # single flowgate
    py -3 src/models/train.py --n-trials 100             # more Optuna trials
    py -3 src/models/train.py --force-cpu                # skip CUDA probe
    py -3 src/models/train.py --retrain                  # overwrite existing models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.cv import MISOTimeSeriesSplit

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_DIR   = Path("data/processed/features")
TARGET_FG_CSV  = Path("data/processed/target_flowgates.csv")
RESULTS_DIR    = Path("data/results")
RESULTS_CSV    = RESULTS_DIR / "training_results.csv"
MODELS_DIR     = Path("models/saved")
IMPORTANCE_DIR = Path("models/importance")

# ── Split boundaries (UTC; MISO EST = UTC-5, fixed, no DST) ────────────────────
_TRAIN_END  = pd.Timestamp("2024-07-01 04:00", tz="UTC")  # 2024-06-30 23:00 EST
_VAL_START  = pd.Timestamp("2024-07-01 05:00", tz="UTC")  # 2024-07-01 00:00 EST
_VAL_END    = pd.Timestamp("2024-10-01 04:00", tz="UTC")  # 2024-09-30 23:00 EST

TARGET_COL       = "binding"
N_ESTIMATORS_MAX = 1_000
EARLY_STOPPING   = 50

_LOADING_COLS = frozenset(["flowgate_loading_pct", "flowgate_loading_pct_is_observed"])
_ID_COLS      = frozenset(["flowgate_id"])

_RESULTS_COLS = [
    "flowgate_id",
    "tier",
    "class_weight_strategy",
    "best_pr_auc",        # val-set PR-AUC (0 when val window is all-negative)
    "optuna_cv_pr_auc",   # mean CV PR-AUC from Optuna tuning on train folds
    "best_params",
    "n_binding_hours_train",
    "training_time_seconds",
    "device_used",
]


# ── Device detection ───────────────────────────────────────────────────────────

# Suppress expected warnings that would otherwise flood output:
#   - XGBoost silently downgrades device="cuda" -> CPU when no GPU is present
#   - XGBoost aucpr metric warns when a CV fold's val set is all one class
#     (expected for highly seasonal constraints in short val windows)
#   - sklearn average_precision_score warns the same way
warnings.filterwarnings(
    "ignore",
    message=r".*(No visible GPU|Device is changed from GPU to CPU).*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Dataset is empty, or contains only positive or negative samples.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*No positive class found in y_true.*",
    category=UserWarning,
)


def _detect_device(force_cpu: bool = False) -> str:
    """
    Return "cuda" only if XGBoost actually uses a GPU.
    Catches XGBoost's silent CPU fallback by checking for its UserWarning.
    """
    if force_cpu:
        print("[INFO] --force-cpu set; using CPU.")
        return "cpu"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            probe = xgb.XGBClassifier(
                n_estimators=1, tree_method="hist", device="cuda"
            )
            probe.fit(
                np.zeros((10, 2), dtype=np.float32),
                np.zeros(10, dtype=np.int8),
            )
        except Exception:
            print("[WARNING] CUDA unavailable — falling back to CPU.")
            return "cpu"

        for w in caught:
            msg = str(w.message)
            if "No visible GPU" in msg or "Device is changed" in msg:
                print("[WARNING] No GPU found — falling back to CPU.")
                return "cpu"

    return "cuda"


# ── Flowgate helpers ───────────────────────────────────────────────────────────

def _safe_id(flowgate_id: str) -> str:
    return flowgate_id.replace("/", "_").replace(" ", "_").replace(".", "_")


def _parquet_path(flowgate_id: str) -> Path:
    return FEATURES_DIR / f"{_safe_id(flowgate_id)}.parquet"


def _load_flowgate_list() -> list[str]:
    """Return ordered list of target flowgate IDs from target_flowgates.csv."""
    if not TARGET_FG_CSV.exists():
        raise FileNotFoundError(
            f"{TARGET_FG_CSV} not found. "
            "Run src/features/build_master_dataset.py first."
        )
    return pd.read_csv(TARGET_FG_CSV, index_col=0).index.tolist()


# ── Class-weight helpers ───────────────────────────────────────────────────────

def get_scale_pos_weight(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return float(n_neg / max(n_pos, 1))


def _class_weight_strategy(ratio: float) -> str:
    if ratio <= 5.0:
        return "none"
    if ratio <= 10.0:
        return "mild"
    return "adjusted"


def _effective_scale_pos_weight(y: pd.Series, strategy: str) -> float:
    if strategy == "none":
        return 1.0
    return get_scale_pos_weight(y)


# ── Flowgate quality tier + feature selection ──────────────────────────────────

def assign_flowgate_tier(binding_rate: float, observed_loading_pct: float) -> str:
    if observed_loading_pct == 0:
        return "synthetic_only"
    if observed_loading_pct < 0.03:
        return "low_signal"
    return "high_signal"


def _select_features(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    to_drop = set(_ID_COLS)
    if tier == "synthetic_only":
        to_drop.update(_LOADING_COLS)
    present = to_drop & set(df.columns)
    return df.drop(columns=list(present))


# ── Date-based split ───────────────────────────────────────────────────────────

def _date_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df). Test rows are simply excluded."""
    train_df = df[df.index <= _TRAIN_END]
    val_df   = df[(df.index >= _VAL_START) & (df.index <= _VAL_END)]
    return train_df, val_df


# ── XGBoost fit ────────────────────────────────────────────────────────────────

def _fit(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    scale_pos_weight: float,
    device: str,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING,
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ── Optuna objective ───────────────────────────────────────────────────────────

def _optuna_objective(
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
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
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


# ── Per-flowgate training pipeline ─────────────────────────────────────────────

def train_flowgate(
    flowgate_id: str,
    features_df: pd.DataFrame,
    n_trials: int,
    device: str,
) -> dict:
    """
    Full training pipeline for one flowgate.

    Returns dict with all tracking columns.
    """
    t_start = time.perf_counter()

    # Tier detection from feature data
    obs_pct = float(
        features_df["flowgate_loading_pct_is_observed"].mean()
        if "flowgate_loading_pct_is_observed" in features_df.columns
        else 0.0
    )
    binding_rate = float(features_df[TARGET_COL].mean())
    tier = assign_flowgate_tier(binding_rate, obs_pct)

    # Feature selection + date split
    selected_df = _select_features(features_df, tier)
    train_df, val_df = _date_split(selected_df)

    # Class-weight strategy (from train labels only)
    y_train = train_df[TARGET_COL]
    ratio    = get_scale_pos_weight(y_train)
    strategy = _class_weight_strategy(ratio)

    # Optuna: CV within train set
    cv = MISOTimeSeriesSplit(n_splits=5, gap_hours=24)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective(trial, train_df, cv, strategy, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # Final fit: train set with val as eval for early stopping
    X_train = train_df.drop(columns=[TARGET_COL])
    X_val   = val_df.drop(columns=[TARGET_COL])
    y_val   = val_df[TARGET_COL]
    final_spw = _effective_scale_pos_weight(y_train, strategy)

    final_model = _fit(
        X_train, y_train,
        X_val,   y_val,
        study.best_params,
        final_spw,
        device,
    )

    # Evaluate on val set
    preds_val = final_model.predict_proba(X_val)[:, 1]
    best_pr_auc = float(average_precision_score(y_val, preds_val))

    elapsed = time.perf_counter() - t_start

    return {
        "flowgate_id":            flowgate_id,
        "tier":                   tier,
        "class_weight_strategy":  strategy,
        "best_pr_auc":            best_pr_auc,
        "optuna_cv_pr_auc":       round(study.best_value, 4),
        "best_params":            json.dumps(study.best_params),
        "n_binding_hours_train":  int(y_train.sum()),
        "training_time_seconds":  round(elapsed, 1),
        "device_used":            device,
        "_model":                 final_model,
        "_feature_names":         list(X_train.columns),
        "_ratio":                 ratio,
    }


# ── Persistence helpers ────────────────────────────────────────────────────────

def _save_model(model: xgb.XGBClassifier, flowgate_id: str) -> Path:
    path = MODELS_DIR / f"{_safe_id(flowgate_id)}.joblib"
    joblib.dump(model, path)
    return path


def _save_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    flowgate_id: str,
) -> None:
    scores = model.feature_importances_
    pd.DataFrame(
        {"feature": feature_names, "importance": scores}
    ).sort_values("importance", ascending=False).to_csv(
        IMPORTANCE_DIR / f"{_safe_id(flowgate_id)}.csv", index=False
    )


def _append_result(result: dict) -> None:
    row = {k: result[k] for k in _RESULTS_COLS}
    row_df = pd.DataFrame([row])
    write_header = not RESULTS_CSV.exists()
    row_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train MISO Layer 1 binding classifiers")
    parser.add_argument("--flowgate",   default=None, help="Train a single flowgate by ID")
    parser.add_argument("--n-trials",   type=int, default=50)
    parser.add_argument("--force-cpu",  action="store_true")
    parser.add_argument("--retrain",    action="store_true",
                        help="Overwrite already-trained models")
    args = parser.parse_args()

    # Directories
    for d in (MODELS_DIR, IMPORTANCE_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Device
    device = _detect_device(args.force_cpu)
    print(f"Device : {device}\n")

    # Flowgate list
    if args.flowgate:
        flowgates = [args.flowgate]
    else:
        flowgates = _load_flowgate_list()

    n_total = len(flowgates)
    n_done  = 0
    n_skip  = 0

    for i, fg_id in enumerate(flowgates, start=1):
        model_path = MODELS_DIR / f"{_safe_id(fg_id)}.joblib"

        if model_path.exists() and not args.retrain:
            n_skip += 1
            print(f"[{i:>3}/{n_total}] {fg_id}  -- SKIP (model exists)")
            continue

        parquet = _parquet_path(fg_id)
        if not parquet.exists():
            print(f"[{i:>3}/{n_total}] {fg_id}  -- SKIP (parquet not found: {parquet})")
            continue

        features_df = pd.read_parquet(parquet)

        try:
            result = train_flowgate(fg_id, features_df, args.n_trials, device)
        except Exception as exc:
            print(f"[{i:>3}/{n_total}] {fg_id}  -- ERROR: {exc}")
            logger.exception("train_flowgate failed for %s", fg_id)
            continue

        _save_model(result["_model"], fg_id)
        _save_importance(result["_model"], result["_feature_names"], fg_id)
        _append_result(result)

        n_done += 1
        cv_flag = " *zero-val*" if result["best_pr_auc"] == 0.0 else ""
        print(
            f"[{i:>3}/{n_total}] {fg_id}"
            f"  val={result['best_pr_auc']:.4f}"
            f"  cv={result['optuna_cv_pr_auc']:.4f}"
            f"  tier: {result['tier']}"
            f"  ratio: {result['_ratio']:.1f}:1"
            f"  strategy: {result['class_weight_strategy']}"
            f"  time: {result['training_time_seconds']:.0f}s"
            f"  device: {device}"
            f"{cv_flag}"
        )

    print(f"\nDone. Trained: {n_done}  Skipped: {n_skip}  Total: {n_total}")
    print(f"Results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
