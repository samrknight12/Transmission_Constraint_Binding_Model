# MISO Congestion Probability Model

## Project Purpose
Binary classification model predicting MISO transmission constraint 
binding events for power trading FTR strategy.

## Stack
- Python 3.11
- XGBoost, LightGBM, scikit-learn, imbalanced-learn
- Optuna for hyperparameter tuning
- SHAP for explainability
- pandas, polars for data wrangling
- pytest for testing

## Key Domain Rules (Never Violate)
- NEVER shuffle time-series data before splitting — use TimeSeriesSplit only
- ALWAYS maintain a 24-hour gap between train and validation folds
- Target variable: shadow_price.abs() > 0.01 = binding (label = 1)
- Class imbalance is ~20:1 — use scale_pos_weight, never raw accuracy
- Primary metric is PR-AUC, not ROC-AUC
- PTDF shift factors are read-only reference data, never features to train on raw

## Data Sources
- MISO Market Reports: https://www.misoenergy.org/markets-and-operations/market-reports
- Binding Constraint Reports: 5-minute interval, columns = [datetime, flowgate_id, shadow_price]
- Outage data: CROW system exports in /data/raw/outages/

## File Conventions
- Feature pipelines return pd.DataFrame with DatetimeIndex
- All timestamps in UTC
- Flowgate IDs follow MISO naming: e.g. "LAKEFIELD.LAKFIELD 345 KV"
- Model artifacts saved to /models/saved/ as .joblib

## Commands
- Run tests: pytest tests/ -v
- Train single flowgate model: python src/models/train.py --flowgate LAKEFIELD
- Evaluate: python src/evaluation/evaluate.py --model-path models/saved/LAKEFIELD.joblib