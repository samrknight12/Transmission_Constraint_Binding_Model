# MISO Transmission Constraint Binding Model

Binary classification model predicting which MISO Day-Ahead transmission
flowgates will bind in a given hour — intended for FTR (Financial Transmission
Right) portfolio strategy.

---

## Purpose

When a transmission constraint binds in the MISO Day-Ahead market, its shadow
price rises above zero, directly affecting congestion component of LMPs across
the system. FTR holders earn (or lose) the shadow price times their MW position.
Predicting which constraints will bind — and when — allows traders to target
high-value FTR paths before the auction.

**Target variable:** `|shadow_price| > 0.01 $/MWh` in the DA binding constraint
report → `binding = 1`. All other hours are `binding = 0`.

---

## Data Sources

Five raw sources covering 2023-01-01 through 2024-12-31 (~2.1 GB on disk):

| Source | Size | What it provides |
|---|---|---|
| DA Binding Constraints | 46 MB | Target variable, binding history features |
| RT Binding Constraints | 140 MB | Flowgate loading % proxy |
| MISO Load Forecast (EIA-930) | ~1 MB | Load level, ramps, % of peak |
| Historical Gen Fuel Mix | 12 MB | Wind/solar forecasts, thermal outage proxy |
| DA Ex-Ante LMP | 799 MB | Downloaded, not yet used in pipeline |

All raw files live under `data/raw/`. CROW outage exports and PTDF shift
factors are MISO-internal (not publicly available); the pipeline degrades
gracefully when either is absent.

Full source documentation, loader details, and format notes:
**[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)**

---

## Master Dataset

`data/processed/master_dataset.parquet`

| Property | Value |
|---|---|
| Shape | 1,929,840 rows x 61 columns |
| Flowgates | 110 (binding rate >= 3% over 2023-2024) |
| Date range | 2023-01-01 – 2024-12-31, hourly |
| Overall binding rate | 7.43% (143,307 positive hours) |
| NaN values | 0 |

**Train / val / test split** (strict chronological, no shuffling):

| Split | EST range | Hours/flowgate | Binding rate |
|---|---|---|---|
| Train | 2023-01-01 – 2024-06-30 | 13,128 | 7.90% |
| Val | 2024-07-01 – 2024-09-30 | 2,208 | 5.98% |
| Test | 2024-10-01 – 2024-12-31 | 2,208 | 6.03% |

60 features across 5 groups: temporal (14), load (10), renewables (10),
outages (8), flowgate-specific (18).

Full column definitions, per-flowgate statistics, flowgate quality tiers,
and class imbalance table:
**[docs/master_dataset.md](docs/master_dataset.md)**

---

## Project Structure

```
src/
  data/
    loaders.py              # load_binding_constraints(), load_load_forecasts(), ...
  features/
    temporal.py             # Hour-of-day, day-of-week, month, season, holiday flags
    load_features.py        # Load level, ramps, rolling averages, % of peak
    renewable_features.py   # Wind/solar forecast MW, penetration %, ramps
    outage_features.py      # Thermal outage MW, % of capacity, proxy flag
    flowgate_features.py    # Loading %, binding history, hours since last binding
    build_features.py       # build_layer1_features() — assembles all 5 groups
    build_master_dataset.py # Orchestrator: all flowgates -> master_dataset.parquet
  models/
    cv.py                   # MISOTimeSeriesSplit (24h gap enforced)
    train.py                # train_flowgate(): Optuna + XGBoost, per-fold scale_pos_weight
  evaluation/
    evaluate.py             # PR-AUC, precision-recall curve, SHAP feature importance
    test_evaluation.py      # Held-out test set evaluation, calibration check, aggregate metrics
    generate_layer1_summary.py  # Produces data/results/layer1_summary.md
  explainability/
    shap_explainer.py       # TreeSHAP drivers, per-hour narratives, daily market summary
  validation/
    sanity_check.py         # PASS/FAIL checks on master_dataset.parquet

scripts/
  retrain_seasonal_features.py  # Retrains zero-val flowgates with 2 seasonal features
  regime_check.py               # Monthly binding plots for regime-change investigation

data/
  raw/                      # Source files (not committed)
  processed/
    master_dataset.parquet  # Full stacked feature matrix
    features/               # Per-flowgate parquets (109 flowgates)
    target_flowgates.csv    # 103 active flowgates (6 dropped as insufficient_signal)
  results/
    training_results.csv    # Per-flowgate training results + model_status + tier
    test_evaluation.csv     # Per-flowgate held-out test metrics and calibration status
    aggregate_metrics.json  # System-level aggregates (top-K precision, medians)
    layer1_summary.md       # Full Layer 1 results report
    daily_explanations/     # Per-day CSVs from explain_day() with market summary header

models/
  saved/                    # Trained .joblib artifacts; *_calibrated.joblib for ECE > 0.05
  shap/
    global/                 # Per-flowgate mean |SHAP| importance CSVs (51 production models)

notebooks/
  outputs/regime_check/     # Monthly binding plots for regime-change flowgates

docs/
  DATA_SOURCES.md           # Raw data sources, loaders, format notes
  master_dataset.md         # Prepared dataset reference, feature tables, flowgate list
```

---

## Layer 1 Results

Full results: **[data/results/layer1_summary.md](data/results/layer1_summary.md)**

109 flowgates trained (103 active, 6 dropped as `insufficient_signal`). 72 models
evaluated on the held-out Oct–Dec 2024 test set.

| Metric | Value |
|---|---|
| Production models (val PR-AUC >= 0.70) | 51 |
| Marginal models (val PR-AUC 0.40-0.69) | 21 |
| Median test PR-AUC — production | 0.8027 |
| Median test PR-AUC — marginal | 0.5514 |
| Top-20 hourly precision | 24.9% |
| Median Brier score | 0.0384 |
| Models requiring calibration (ECE > 0.05) | 32 / 51 |

**Tier breakdown** (models with >= 1 test binding hour):

| Tier | Count | Median test PR-AUC |
|---|---|---|
| high_signal | 21 | 0.8339 |
| low_signal | 24 | 0.7905 |
| synthetic_only | 18 | 0.6210 |

4 flowgates confirmed as regime-change (network reconfiguration Q1 2024; excluded from
test evaluation). 18 zero-val flowgates retrained with seasonal features
(`binding_rate_same_month_prior_year`, `days_since_last_binding_rolling_14d`).

---

## Explainability

`src/explainability/shap_explainer.py` provides four levels of model explanation for
production models, using XGBoost's native `pred_contribs` (TreeSHAP) to avoid library
version incompatibilities.

**Global importance** (`run_global_importance_all`) — mean |SHAP| per feature across
the full Oct–Dec 2024 test set, saved per flowgate to `models/shap/global/`. The most
dominant feature across 50 of 51 production models is `flowgate_hours_since_binding`.

**Single-hour explanation** (`explain_prediction(flowgate_id, datetime_utc)`) — returns
predicted probability (calibrated where available), top drivers with SHAP value and
human-readable value context, historical base rate, and direction vs base rate.

**Trader narrative** (`generate_narrative(flowgate_id, datetime_utc)`) — plain-English
3-sentence output with no ML terminology. Binding-history features
(`hours_since_last_binding`, `binding_freq_trailing_7d/30d`) are intentionally suppressed
from the narrative text (they drive the model but are not actionable for a trader);
if all top drivers are excluded, a generic "recent binding activity" fallback fires.

**Daily batch** (`explain_day(date, top_n_flowgates=10)`) — runs every production model
across all hours of a given day, filters to above-threshold predictions, keeps top-N
per hour, deduplicates consecutive hours where a flowgate holds identical probability
for 3+ hours (synthetic loading proxy signal), and generates a market summary. Returns
`(DataFrame, summary_str)` and saves to `data/results/daily_explanations/{date}.csv`
with the summary as a leading comment row.

```
# October 15 congestion summary: 8 constraints flagged above threshold.
# Loading vs 30-day max is the dominant driver across LIMECK-BARTON and
# FARGO-SHEYN (+3 others). Wind ramp rate also contributes across 4 constraints.
# MUDLAKE-VERONA 1 remains elevated across 6 output rows driven by solar
# generation forecast.
datetime,flowgate_id,probability,narrative
01:00-05:00 UTC,LIMECK-BARTON FLO KILLDEER-QUINN,0.731,...[5hrs collapsed]
2024-10-15 02:00+00:00,FARGO-SHEYN FLO CTR-JAMESTOWN 345,0.931,...
03:00-10:00 UTC,ASTORIA TR1_TR11 FLO BRKINGS CNTY-ASTRIA,0.826,...[8hrs collapsed]
```

Example narrative (Oct 15 14:00 UTC, MUDLAKE-VERONA 1):
> *At 14:00 UTC on 2024-10-15, MUDLAKE-VERONA 1 FLO MUDLAKE-VERONA 2 has a 95%
> binding probability -- 91 percentage points above its historical base rate of 4%.
> Solar generation forecast (3,647 MW -- 279% above avg) is the primary driver,
> pushing probability upward; followed by solar ramp rate (+1,492 MW) pulling
> downward. Additionally, wind forecast error (+4,134 MW) contributes to suppressed
> risk.*

**Design notes:**
- Probabilities clipped to `(0.001, 0.999)` at both the base model and calibrator
  output — isotonic regression can otherwise saturate at exactly 1.0
- Binding-history features are excluded from narrative text but remain in the model;
  the suppression list is `EXCLUDED_FROM_NARRATIVE` in `shap_explainer.py`
- Consecutive-run collapse threshold is 3+ hours at identical probability (3 dp);
  tagged `[synthetic loading proxy active]` in the narrative

---

## Current Progress

| Stage | Status |
|---|---|
| Data ingestion (`src/data/loaders.py`) | Complete |
| Feature engineering (`src/features/`) | Complete — 59 features, 5 groups |
| Master dataset build (`build_master_dataset.py`) | Complete — 109 flowgates, 0 NaN |
| Leakage guards | Complete — shift(1) verified, corr gate, leakage assertions |
| Sanity checks (`sanity_check.py`) | Complete — 11/13 pass (2 by-design) |
| Flowgate quality tiers | Complete — synthetic_only / low_signal / high_signal |
| Training pipeline (`train.py`) | Complete — Optuna, per-fold scale_pos_weight, tier-based feature selection |
| Layer 1 model training | Complete — 103 active flowgates trained |
| Seasonal feature retraining | Complete — 18 zero-val flowgates retrained with 2 seasonal features |
| Regime change analysis | Complete — 4 flowgates confirmed, monthly binding plots saved |
| Test set evaluation (`test_evaluation.py`) | Complete — per-flowgate metrics, calibration check |
| Calibration | Complete — isotonic recalibration for 32 production models; `*_calibrated.joblib` saved |
| Layer 1 summary | Complete — `data/results/layer1_summary.md` |
| Explainability (`shap_explainer.py`) | Complete — global importance, per-hour narratives, daily market summary |
| Layer 2 (signal aggregation / ensemble) | Not started |
| Layer 3 (FTR strategy / backtesting) | Not started |

Sanity check failures are expected by design:

- **Class imbalance check**: High-activity flowgates (e.g. CHAR_CK, >20% binding) fall
  below the 10:1 target ratio. The `none` class-weight strategy handles these correctly.
- **Observed loading rate**: 29 synthetic_only flowgates have zero RT loading observations;
  44 low_signal flowgates have sparse coverage. The tier system was built for this.

---

## Key Commands

```bash
# Build master dataset
py -3 src/features/build_master_dataset.py

# Run sanity checks on master dataset
py -3 src/validation/sanity_check.py

# Train a single flowgate
py -3 src/models/train.py \
    --flowgate "LAKEFIELD.LAKFIELD 345 KV" \
    --features-path data/processed/features/LAKEFIELD_LAKFIELD_345_KV.parquet

# Retrain zero-val flowgates with seasonal features (n_trials=75)
py -3 scripts/retrain_seasonal_features.py
py -3 scripts/retrain_seasonal_features.py --min-cv 0.50 --n-trials 100

# Run held-out test evaluation + calibration check
py -3 src/evaluation/test_evaluation.py

# Generate Layer 1 summary report (run test_evaluation.py first)
py -3 src/evaluation/generate_layer1_summary.py

# Monthly binding plots for regime-change investigation
py -3 scripts/regime_check.py

# Compute global SHAP importance for all 51 production models
py -3 src/explainability/shap_explainer.py

# Run daily explanation for a specific date
py -3 src/explainability/shap_explainer.py --day 2024-10-15

# Use explainability API directly
python -c "
from src.explainability.shap_explainer import explain_prediction, generate_narrative, explain_day
print(generate_narrative('FARGO-SHEYN FLO CTR-JAMESTOWN 345', '2024-10-15 05:00'))
result, summary = explain_day('2024-10-15', top_n_flowgates=10)
"

# Run tests
pytest tests/ -v
```

---

## Domain Invariants

- **Never shuffle time-series data** — use `MISOTimeSeriesSplit` only
- **24-hour gap** between train and validation folds (MISO DA market closes 24h ahead)
- **Primary metric: PR-AUC** — ROC-AUC is misleading at 20:1 class imbalance
- **scale_pos_weight computed from training fold only** — never from validation or test
- **PTDF shift factors are reference data** — never used as raw model features
- All timestamps in UTC internally; MISO market uses fixed EST (UTC-5, no DST)
