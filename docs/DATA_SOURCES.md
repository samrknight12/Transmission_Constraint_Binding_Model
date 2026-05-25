# Data Sources

All data lives under `data/raw/`. Coverage period: **2023-01-01 through 2024-12-31 (UTC)**.

---

## 1. DA Binding Constraints — `data/raw/binding_constraints/`

| Property | Value |
|---|---|
| Files | `2023_da_bc_HIST.csv`, `2024_da_bc_HIST.csv` |
| Size | 46 MB (2 annual files) |
| Source | MISO Market Reports — public |
| URL pattern | `https://docs.misoenergy.org/marketreports/{year}_da_bc_HIST.csv` |
| Loader | `src/data/loaders.py::load_binding_constraints()` |

**What it contains:** Every hour in which a transmission flowgate was binding in the Day-Ahead market — the constraint's shadow price ($/MWh), constraint name, branch description, and contingency. Non-binding hours are absent (shadow price implicitly 0).

**How it is used:**
- Target variable: `|shadow_price| > 0.01 $/MWh` → `binding = 1`
- Historical binding frequency features (lagged rolling windows)
- Hours-since-last-binding feature
- Complete hourly time index is reconstructed by reindexing to `load_forecast_df.index`; missing hours receive `shadow_price = 0`

**Format notes:** Two header rows, `shadow_price` strings use MISO parenthesis notation for negatives (`($3.76)` → −3.76), timestamps in EST HourEnding 1–24 convention converted to UTC on load.

---

## 2. RT Binding Constraints — `data/raw/rt_bc/`

| Property | Value |
|---|---|
| Files | `2023_rt_bc_HIST.csv`, `2024_rt_bc_HIST.csv` |
| Size | 140 MB (2 annual files) |
| Source | MISO Market Reports — public |
| URL pattern | `https://docs.misoenergy.org/marketreports/{year}_rt_bc_HIST.csv` |
| Loader | None (pre-processed into flowgate_loading at build time) |

**What it contains:** Every 5-minute real-time interval in which a flowgate was binding, with HH:MM timestamps in EST and a preliminary shadow price.

**How it is used:** Aggregated to hourly resolution to construct a `loading_pct` proxy for `data/raw/flowgate_loading/`. The number of binding 5-minute intervals per hour (out of a maximum of 12) maps to `loading_pct = 90 + 10 × (binding_intervals / 12)`. Constraint names in this file use a different naming convention from the DA file — they are linked back to DA flowgate names via `Constraint_ID`, which is shared between both files (~97% match rate).

---

## 3. MISO Load Forecast — `data/raw/load_forecasts/`

| Property | Value |
|---|---|
| File (primary) | `eia930_miso_demand.csv` (0.8 MB) |
| Files (legacy) | `DA_Load_EPNodes_YYYYMMDD.zip` × 731 (1,064 MB) |
| Source | EIA Form 930 Grid Monitor — public |
| URL | `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{half}.csv` |
| Loader | `src/data/loaders.py::load_load_forecasts()` |

**What it contains (EIA-930):** Hourly MISO balancing-authority demand forecast (MW) as published one day ahead. Range: 53,933–127,807 MW, covering all of MISO.

**How it is used:** All load features in `src/features/load_features.py` — absolute load level, ramp rates, deviation from rolling average, percent of peak. Also used as the denominator for `renewable_penetration_pct`.

**Loader priority:** The loader checks for `eia930_miso_demand.csv` first. EIA-930 timestamps are end-of-hour UTC; the loader subtracts 1 hour on load to align with MISO's start-of-hour convention. 24 hours of missing data on 2024-07-02 are forward-filled.

**Legacy files (not used):** The 731 `DA_Load_EPNodes_*.zip` files contain EPNode-level **LMP prices** ($/MWh), not load in MW. Their sum reaches 3.4 million, making them unsuitable as a load proxy. They are retained in case a specific LMP signal is needed in future features but are not read by any current loader.

---

## 4. Historical Generation Fuel Mix — `data/raw/gen_fuel_mix/`

| Property | Value |
|---|---|
| Files | `historical_gen_fuel_mix_2023.xlsx`, `historical_gen_fuel_mix_2024.xlsx` |
| Size | 12 MB (2 annual files) |
| Source | MISO Market Reports — public |
| URL pattern | `https://docs.misoenergy.org/marketreports/historical_gen_fuel_mix_{year}.xlsx` |
| Loaders | `load_gen_fuel_mix()`, `load_renewables()`, `load_outages()` |

**What it contains:** Hourly DA cleared and RT actual generation (MW) by fuel type (Coal, Gas, Hydro, Nuclear, Other, Solar, Storage, Wind) across all MISO regions.

**How it is used for three separate purposes:**

1. **Wind and solar features** (`load_renewables()`): DA Cleared UDS Generation for Wind and Solar fuel types is summed across all regions to produce `wind_forecast_mw` and `solar_forecast_mw`. DA Cleared is the quantity that entered the DA market schedule and equals the DA forecast for these resources (wind/solar submit $0 bids and are dispatched at their forecast output). RT Generation State Estimator gives actuals.

2. **Outage proxy** (`load_outages()` in proxy mode): Coal + Gas + Nuclear DA cleared generation summed across regions gives a thermal generation level. Sustained drops below the 30-day rolling mean signal planned or forced outages. This proxy activates only when no CROW outage files are present in `data/raw/outages/`.

3. **Raw fuel mix access** (`load_gen_fuel_mix()`): Full long-format table available for ad hoc analysis.

---

## 5. DA Ex-Ante LMP — `data/raw/lmp/`

| Property | Value |
|---|---|
| Files | `YYYYMMDD_da_exante_lmp.csv` × 731 |
| Size | 799 MB |
| Source | MISO Market Reports — public |
| URL pattern | `https://docs.misoenergy.org/marketreports/{YYYYMMDD}_da_exante_lmp.csv` |
| Loader | None (not used in current feature pipeline) |

**What it contains:** Day-ahead ex-ante LMP prices ($/MWh) at every MISO node, hub, load zone, and interface for all 24 hours, broken down into LMP, MCC (congestion), and MLC (loss) components.

**Current status:** Downloaded and available but not consumed by any loader in the current pipeline. Interface-level MCC values could be used in a future feature to derive interface congestion signals or to approximate flowgate loading direction.

---

## 6. Flowgate Loading — `data/raw/flowgate_loading/`

| Property | Value |
|---|---|
| File | `rt_binding_loading_2023_2024.csv` |
| Size | 7.9 MB |
| Source | Derived — computed from RT Binding Constraint HIST files |
| Loader | `src/data/loaders.py::load_flowgate_loading()` |

**What it contains:** Hourly `loading_pct` estimates for 1,440 DA flowgates (of 3,581 total). Values range from 90.8 to 100.0.

**How it is derived:** For each (UTC hour, flowgate) pair, counts the number of binding 5-minute RT intervals (max 12 per hour). Maps to `loading_pct = 90 + 10 × (count / 12)`. Only hours with at least one binding interval appear; non-binding hours are absent from the file.

**Important limitation:** This is a synthetic proxy, not actual MW flow data. MISO does not publish historical flowgate loading percentages publicly. For the 2,141 DA flowgates with no RT binding events, all hours default to `loading_pct = 85.0` via `fillna` in `flowgate_features.py`. The loading signal is most reliable for highly and repeatedly binding constraints.

---

## Summary

| Directory | Size | Status | Features produced |
|---|---|---|---|
| `binding_constraints/` | 46 MB | Complete | Target variable, binding history |
| `rt_bc/` | 140 MB | Complete | Flowgate loading proxy |
| `load_forecasts/` | 1,064 MB | Complete (EIA-930 active) | Load level, ramps, % of peak |
| `gen_fuel_mix/` | 12 MB | Complete | Wind/solar forecast, outage proxy |
| `lmp/` | 799 MB | Downloaded, unused | (Future: interface congestion) |
| `flowgate_loading/` | 8 MB | Synthetic proxy | Loading %, distance to limit |

**Total raw data on disk: ~2.1 GB**

**Note on unobtainable data:** CROW outage exports and PTDF shift factors are not publicly available (MISO internal systems only). When no CROW files are present, `load_outages()` automatically falls back to the gen_fuel_mix thermal proxy with an `_is_proxy=True` flag. `load_ptdf()` returns an empty DataFrame. Both pipeline paths are production-ready.
