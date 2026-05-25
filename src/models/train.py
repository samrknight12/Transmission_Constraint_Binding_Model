"""
Training entrypoint for MISO Layer 1 binding classifier.

Usage:
    python src/models/train.py --flowgate LAKEFIELD --features-path data/processed/LAKEFIELD.parquet

Pipeline:
  1. Load pre-built feature parquet (output of build_layer1_features).
  2. Determine per-flowgate class-weighting strategy from training data.
  3. Tune XGBoost hyperparameters with Optuna (TimeSeriesSplit, PR-AUC objective).
  4. Refit on full dataset with best params.
  5. Save artifact to models/saved/<flowgate_id>.joblib.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.models.cv import MISOTimeSeriesSplit

logger = logging.getLogger(__name__)

TARGET_COL       = "binding"
MODELS_DIR       = Path("models/saved")
N_ESTIMATORS_MAX = 1_000
EARLY_STOPPING   = 50


# ── Class-weight helpers ──────────────────────────────────────────────────────

def get_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Compute the raw neg/pos imbalance ratio from the training fold.

    Call this on training data only — never on validation or test data,
    which would leak future class-distribution information into the model.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return float(n_neg / max(n_pos, 1))


def _class_weight_strategy(ratio: float) -> str:
    """
    Map an imbalance ratio to a named class-weighting strategy.

    "adjusted"  ratio >  10:1  — full scale_pos_weight applied
    "mild"      5:1 < ratio <= 10:1  — full scale_pos_weight applied
    "none"      ratio <= 5:1  — scale_pos_weight forced to 1.0; the class
                distribution is close enough that XGBoost's default loss handles
                it without aggressive reweighting (avoids over-correcting on
                highly active constraints like CHAR_CK that bind >20% of hours)
    """
    if ratio <= 5.0:
        return "none"
    if ratio <= 10.0:
        return "mild"
    return "adjusted"


def _effective_scale_pos_weight(y_train: pd.Series, strategy: str) -> float:
    """Return the scale_pos_weight value to pass to XGBoost for this fold/fit."""
    if strategy == "none":
        return 1.0
    return get_scale_pos_weight(y_train)


# ── Flowgate quality tier and feature selection ───────────────────────────────

def assign_flowgate_tier(binding_rate: float, observed_loading_pct: float) -> str:
    """
    Classify a flowgate by the reliability of its loading features.

    "synthetic_only"  observed_loading_pct == 0   — all loading_pct values are
                      the 85.0 fill; no RT confirmation exists at any hour.
    "low_signal"      0 < observed_loading_pct < 3% — sparse RT confirmation.
    "high_signal"     observed_loading_pct >= 3%  — reliable loading features.
    """
    if observed_loading_pct == 0:
        return "synthetic_only"
    if observed_loading_pct < 0.03:
        return "low_signal"
    return "high_signal"


# Columns carrying no signal for flowgates whose loading is entirely synthetic.
_LOADING_COLS = frozenset(["flowgate_loading_pct", "flowgate_loading_pct_is_observed"])
# String identifier — never a model input.
_ID_COLS = frozenset(["flowgate_id"])


def _select_features(features_df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """
    Return features_df with non-model columns removed for the given tier.

    synthetic_only : drop flowgate_loading_pct and flowgate_loading_pct_is_observed.
                     Both are constant (85.0 / 0) for these flowgates and add noise.
    low_signal     : keep flowgate_loading_pct_is_observed so the model learns to
                     discount hours where loading is synthetic fill.
    high_signal    : use all features as-is.

    flowgate_id is always dropped (string identifier, not a model input).
    """
    to_drop = set(_ID_COLS)
    if tier == "synthetic_only":
        to_drop.update(_LOADING_COLS)
    present = to_drop & set(features_df.columns)
    return features_df.drop(columns=list(present))


# ── Model training ────────────────────────────────────────────────────────────

def _train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    scale_pos_weight: float,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _optuna_objective(
    trial: optuna.Trial,
    features_df: pd.DataFrame,
    cv: MISOTimeSeriesSplit,
    strategy: str,
) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    pr_aucs: list[float] = []
    for train_df, val_df in cv.split_by_date(features_df):
        X_tr = train_df.drop(columns=[TARGET_COL])
        y_tr = train_df[TARGET_COL]
        X_v  = val_df.drop(columns=[TARGET_COL])
        y_v  = val_df[TARGET_COL]

        # scale_pos_weight recomputed from this fold's training labels only
        spw = _effective_scale_pos_weight(y_tr, strategy)
        model = _train_fold(X_tr, y_tr, X_v, y_v, params, scale_pos_weight=spw)
        preds = model.predict_proba(X_v)[:, 1]
        pr_aucs.append(average_precision_score(y_v, preds))

    return float(np.mean(pr_aucs))


def train_flowgate(
    flowgate_id: str,
    features_df: pd.DataFrame,
    n_trials: int = 50,
) -> dict:
    """
    Full training pipeline for one flowgate.

    1. Compute imbalance ratio and assign class-weighting strategy.
    2. Optuna tuning (TimeSeriesSplit, PR-AUC), applying strategy per fold.
    3. Refit on full dataset with best params.
    4. Save to models/saved/<flowgate_id>.joblib.

    Returns
    -------
    dict with keys:
        flowgate_id, model_path, best_pr_auc,
        imbalance_ratio, class_weight_strategy
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tier: determines which loading columns are kept ───────────────────────
    observed_loading_pct = float(
        features_df["flowgate_loading_pct_is_observed"].mean()
        if "flowgate_loading_pct_is_observed" in features_df.columns
        else 0.0
    )
    binding_rate = float(features_df[TARGET_COL].mean())
    tier = assign_flowgate_tier(binding_rate, observed_loading_pct)

    # Apply tier-based column selection once; both Optuna and final fit use the result.
    selected_df = _select_features(features_df, tier)

    # ── Class-weighting strategy ──────────────────────────────────────────────
    y_full = selected_df[TARGET_COL]
    ratio = get_scale_pos_weight(y_full)
    strategy = _class_weight_strategy(ratio)

    print(
        f"  {flowgate_id}"
        f"  tier={tier}"
        f"  obs_loading={observed_loading_pct:.2%}"
        f"  imbalance={ratio:.1f}:1"
        f"  strategy={strategy}"
    )

    cv = MISOTimeSeriesSplit(n_splits=5, gap_hours=24)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _optuna_objective(trial, selected_df, cv, strategy),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(
        "Best PR-AUC: %.4f | params: %s",
        study.best_value,
        study.best_params,
    )

    X = selected_df.drop(columns=[TARGET_COL])
    y = selected_df[TARGET_COL]
    final_spw = _effective_scale_pos_weight(y, strategy)

    final_model = xgb.XGBClassifier(
        **study.best_params,
        scale_pos_weight=final_spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS_MAX,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    safe_id  = flowgate_id.replace(" ", "_").replace("/", "_")
    out_path = MODELS_DIR / f"{safe_id}.joblib"
    joblib.dump(final_model, out_path)
    logger.info("Saved model -> %s", out_path)

    return {
        "flowgate_id":           flowgate_id,
        "model_path":            out_path,
        "best_pr_auc":           study.best_value,
        "imbalance_ratio":       ratio,
        "class_weight_strategy": strategy,
        "tier":                  tier,
        "n_features":            X.shape[1],
    }


def _print_tier_counts(fg_csv: Path) -> None:
    """Print tier distribution from target_flowgates.csv if it exists."""
    if not fg_csv.exists() or not fg_csv.stat().st_size:
        return
    fg_df = pd.read_csv(fg_csv, index_col=0)
    if "tier" not in fg_df.columns:
        return
    counts = fg_df["tier"].value_counts()
    print("Flowgate tier counts (across all target flowgates):")
    for tier_name, cnt in counts.items():
        print(f"  {tier_name:<20s}: {cnt:>3d}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train MISO Layer 1 binding classifier")
    parser.add_argument("--flowgate",      required=True, help="MISO flowgate ID")
    parser.add_argument("--features-path", required=True, help="Parquet file from build_features")
    parser.add_argument("--n-trials",      type=int, default=50)
    args = parser.parse_args()

    _print_tier_counts(Path("data/processed/target_flowgates.csv"))

    features = pd.read_parquet(args.features_path)
    result   = train_flowgate(args.flowgate, features, n_trials=args.n_trials)
    logger.info(
        "Done | tier=%s | strategy=%s | PR-AUC=%.4f | features=%d | model=%s",
        result["tier"],
        result["class_weight_strategy"],
        result["best_pr_auc"],
        result["n_features"],
        result["model_path"],
    )


if __name__ == "__main__":
    main()
