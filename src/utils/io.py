"""
Model artifact I/O — save and load .joblib files from models/saved/.
"""
from __future__ import annotations

from pathlib import Path

import joblib

MODELS_DIR = Path("models/saved")


def _safe_id(flowgate_id: str) -> str:
    return flowgate_id.replace(" ", "_").replace("/", "_")


def save_model(model, flowgate_id: str) -> Path:
    """Save a trained model to models/saved/<flowgate_id>.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{_safe_id(flowgate_id)}.joblib"
    joblib.dump(model, path)
    return path


def load_model(flowgate_id: str):
    """Load models/saved/<flowgate_id>.joblib."""
    path = MODELS_DIR / f"{_safe_id(flowgate_id)}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for flowgate '{flowgate_id}' at {path}")
    return joblib.load(path)
