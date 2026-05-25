"""
Model evaluation for MISO Layer 1 binding classifier.

Primary metric: PR-AUC (average_precision_score).
ROC-AUC included for reference but NOT used for model selection — it is
misleading under 20:1 class imbalance.

Usage:
    python src/evaluation/evaluate.py \
        --model-path models/saved/LAKEFIELD.joblib \
        --features-path data/processed/LAKEFIELD.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TARGET_COL = "binding"
DECISION_THRESHOLD = 0.5


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    flowgate_id: str = "",
    output_dir: Path = Path("models/evaluation"),
) -> dict[str, float]:
    """
    Compute evaluation metrics and write diagnostic plots.

    Returns a dict with pr_auc, roc_auc, positive_rate, n_total.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= DECISION_THRESHOLD).astype(int)

    pr_auc  = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)

    logger.info("[%s] PR-AUC: %.4f | ROC-AUC: %.4f (reference only)", flowgate_id, pr_auc, roc_auc)
    logger.info("\n%s", classification_report(y, pred))

    # ── Precision-Recall curve ────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y, proba)
    baseline = float(y.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(baseline, linestyle="--", color="grey",
               label=f"Random baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {flowgate_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{flowgate_id}_pr_curve.png", dpi=150)
    plt.close(fig)

    # ── SHAP feature importance ───────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.savefig(output_dir / f"{flowgate_id}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "pr_auc":        pr_auc,
        "roc_auc":       roc_auc,
        "positive_rate": baseline,
        "n_total":       len(y),
        "n_positive":    int(y.sum()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate MISO binding classifier")
    parser.add_argument("--model-path",    required=True)
    parser.add_argument("--features-path", required=True)
    args = parser.parse_args()

    model    = joblib.load(args.model_path)
    features = pd.read_parquet(args.features_path)

    X = features.drop(columns=[TARGET_COL])
    y = features[TARGET_COL]
    flowgate_id = Path(args.model_path).stem.replace("_", " ")

    evaluate_model(model, X, y, flowgate_id=flowgate_id)


if __name__ == "__main__":
    main()
