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
  validation/
    sanity_check.py         # 6 PASS/FAIL checks on master_dataset.parquet

data/
  raw/                      # Source files (not committed)
  processed/
    master_dataset.parquet  # Full stacked feature matrix
    features/               # Per-flowgate parquets
    target_flowgates.csv    # 110 qualifying flowgates with binding rates and tiers

models/
  saved/                    # Trained .joblib artifacts

docs/
  DATA_SOURCES.md           # Raw data sources, loaders, format notes
  master_dataset.md         # Prepared dataset reference, feature tables, flowgate list
```

---

## Current Progress

| Stage | Status |
|---|---|
| Data ingestion (`src/data/loaders.py`) | Complete |
| Feature engineering (`src/features/`) | Complete — 60 features, 5 groups |
| Master dataset build (`build_master_dataset.py`) | Complete — 110 flowgates, 0 NaN |
| Leakage guards | Complete — shift(1) verified, corr gate, leakage assertions |
| Sanity checks (`sanity_check.py`) | Complete — 11/13 checks pass (2 by-design) |
| Flowgate quality tiers | Complete — synthetic_only / low_signal / high_signal |
| Training pipeline (`train.py`) | Complete — Optuna tuning, per-fold scale_pos_weight, tier-based feature selection |
| Model evaluation (`evaluate.py`) | Scaffolded — PR-AUC, PR curve, SHAP |
| Model training (running fits) | Not started |
| Backtesting / FTR strategy layer | Not started |

The two sanity check failures are expected by design:

- **Check 2** (class imbalance 10-35:1): High-activity flowgates like CHAR_CK
  bind >20% of hours; their imbalance ratio falls below 10:1. The "none"
  class-weight strategy handles these.
- **Check 6** (observed loading rate 3-10%): 29 flowgates have zero RT loading
  observations (synthetic_only tier); 44 have sparse coverage (low_signal tier).
  The tier system was built specifically to handle these cases.

---

## Key Commands

```bash
# Build master dataset (all 110 flowgates)
py -3 src/features/build_master_dataset.py

# Run sanity checks on master dataset
py -3 src/validation/sanity_check.py

# Train a single flowgate
py -3 src/models/train.py \
    --flowgate "LAKEFIELD.LAKFIELD 345 KV" \
    --features-path data/processed/features/LAKEFIELD_LAKFIELD_345_KV.parquet

# Evaluate a trained model
py -3 src/evaluation/evaluate.py \
    --model-path models/saved/LAKEFIELD_LAKFIELD_345_KV.joblib \
    --features-path data/processed/features/LAKEFIELD_LAKFIELD_345_KV.parquet

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
