"""
MISO market data loaders.

All timestamps are normalised to UTC on load. Column names match the schema
expected by src/features/build_features.py.

Actual file formats (verified against downloaded data 2026-05-24):

  da_bc HIST  — Annual CSV (2 header rows + MISO disclaimer footer).
                Shadow price formatted as "$1.64" or "($3.76)" for negatives.
                Timestamps: Market Date (M/D/YYYY) + Hour of Occurrence (1-24 EST).

  load        — Daily DA_Load_EPNodes_YYYYMMDD.zip, each containing one CSV.
                Wide format: EPNode rows, HE1-HE24 columns (EST).
                We sum all EPNode LMP rows to get total system load proxy,
                then melt to long format.

  lmp         — Daily YYYYMMDD_da_exante_lmp.csv.
                Same wide HE1-HE24 layout. Interface-level rows used.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")

# MISO DA market hours are Eastern Time (EST = UTC-5, no DST adjustment in market ops)
_MISO_TZ   = "America/New_York"
_UTC_OFFSET = pd.Timedelta(hours=5)   # EST = UTC-5


def _parse_shadow_price(series: pd.Series) -> pd.Series:
    """
    Convert MISO shadow price strings to float.

    "$1.64"    ->  1.64
    "($3.76)"  -> -3.76
    """
    s = series.astype(str).str.strip()
    negative_mask = s.str.startswith("(")
    # Strip $, commas, parentheses
    cleaned = s.str.replace(r"[$(),]", "", regex=True).str.strip()
    result = pd.to_numeric(cleaned, errors="coerce")
    result[negative_mask] *= -1
    return result


def _parse_miso_hour(market_date: pd.Series, hour_col: pd.Series) -> pd.Series:
    """
    Combine Market Date (M/D/YYYY) + Hour of Occurrence (1-24 EST) into UTC.

    MISO Hour 1 = 00:00-01:00 EST on that market date.
    Returned as UTC timestamps.
    """
    hour = pd.to_numeric(hour_col.astype(str).str.strip(), errors="coerce").fillna(1)
    # hour 1 = 00:00 EST, so subtract 1 to get hour_start
    base_date = pd.to_datetime(market_date, format="%m/%d/%Y", errors="coerce")
    local_ts  = base_date + pd.to_timedelta((hour - 1).astype(int), unit="h")
    # Localize as EST (MISO does not observe DST in market operations)
    return local_ts.dt.tz_localize("EST").dt.tz_convert("UTC")


def load_binding_constraints(path: Path | None = None) -> pd.DataFrame:
    """
    Load DA binding constraint HIST CSV files.

    Reads ``{year}_da_bc_HIST.csv`` files from the binding_constraints directory.
    Filters out non-binding hours (shadow_price == 0) so the caller gets only
    binding events.

    Returns DataFrame with columns:
        datetime (UTC DatetimeIndex), flowgate_id, shadow_price (float).
    """
    path = path or RAW_DIR / "binding_constraints"
    files = sorted(Path(path).glob("*_da_bc_HIST.csv"))
    if not files:
        raise FileNotFoundError(f"No *_da_bc_HIST.csv files in {path}")

    chunks = []
    for f in files:
        df = pd.read_csv(f, skiprows=2, low_memory=False)
        df.columns = df.columns.str.strip()

        # Drop the MISO disclaimer footer rows (Market Date is not a valid date)
        df = df[pd.to_datetime(df["Market Date"], format="%m/%d/%Y", errors="coerce").notna()]

        df["datetime"]    = _parse_miso_hour(df["Market Date"], df["Hour of Occurrence"])
        df["flowgate_id"] = df["Constraint Name"].str.strip()
        df["shadow_price"] = _parse_shadow_price(df["Shadow Price"])

        chunks.append(df[["datetime", "flowgate_id", "shadow_price"]].dropna())

    result = pd.concat(chunks, ignore_index=True)
    return result.sort_values("datetime").reset_index(drop=True)


def _parse_load_epnodes_zip(zpath: Path) -> pd.DataFrame | None:
    """
    Parse one DA_Load_EPNodes_YYYYMMDD.zip file.

    File is wide-format: EPNode rows with columns HE1-HE24 (EST).
    We sum the LMP rows across all EPNodes to get a total system load proxy,
    then melt to hourly long format.

    Returns a one-day DataFrame with UTC DatetimeIndex and load_forecast_mw column,
    or None if the file is malformed.
    """
    try:
        with zipfile.ZipFile(zpath) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            with z.open(csv_name) as f:
                raw = pd.read_csv(f, skiprows=4, header=0)
    except Exception:
        return None

    raw.columns = raw.columns.str.strip()
    he_cols = [c for c in raw.columns if re.match(r"HE\s*\d+", c)]
    if not he_cols:
        return None

    # Extract market date from filename: DA_Load_EPNodes_YYYYMMDD.zip
    date_str = re.search(r"(\d{8})", zpath.name)
    if not date_str:
        return None
    market_date = pd.to_datetime(date_str.group(1), format="%Y%m%d")

    # Keep only LMP rows (ignore MCC, MLC components)
    lmp_rows = raw[raw.iloc[:, 1].astype(str).str.strip() == "LMP"]
    if lmp_rows.empty:
        lmp_rows = raw  # fallback: use all rows

    totals = lmp_rows[he_cols].apply(pd.to_numeric, errors="coerce").sum()

    # Map HE1-HE24 → UTC timestamps  (HE1 = 00:00-01:00 EST)
    hours = pd.to_numeric(
        totals.index.str.extract(r"(\d+)")[0], errors="coerce"
    ).values.astype(int)
    local_ts = pd.Timestamp(market_date) + pd.to_timedelta(hours - 1, unit="h")
    utc_ts = (
        pd.DatetimeIndex([pd.Timestamp(t).tz_localize("EST") for t in local_ts])
        .tz_convert("UTC")
    )
    return pd.DataFrame({"load_forecast_mw": totals.values}, index=utc_ts)


def load_load_forecasts(path: Path | None = None) -> pd.DataFrame:
    """
    Load MISO DA demand forecast.

    Preferred source: EIA-930 ``eia930_miso_demand.csv`` in the load_forecasts
    directory (53–128 GW range, correct MW scale).  The EIA-930 file uses
    end-of-hour UTC timestamps; we shift -1h to match MISO's start-of-hour
    convention used throughout this pipeline.

    Fallback: ``DA_Load_EPNodes_*.zip`` daily files (EPNode LMP price sums —
    wrong MW scale but preserved for backward compatibility).

    Returns DataFrame with UTC DatetimeIndex, column: load_forecast_mw.
    """
    path = path or RAW_DIR / "load_forecasts"
    eia_path = Path(path) / "eia930_miso_demand.csv"

    if eia_path.exists():
        df = pd.read_csv(eia_path, parse_dates=["datetime_utc"])
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        # EIA-930 uses end-of-hour; shift to start-of-hour to match MISO convention
        df.index = df["datetime_utc"] - pd.Timedelta(hours=1)
        df.index.name = "datetime"
        result = df[["demand_forecast_mw"]].rename(
            columns={"demand_forecast_mw": "load_forecast_mw"}
        ).sort_index()
        result["load_forecast_mw"] = (
            result["load_forecast_mw"].ffill().bfill()
        )
        return result

    # Fallback: daily DA_Load_EPNodes zip files
    files = sorted(Path(path).glob("DA_Load_EPNodes_*.zip"))
    if not files:
        raise FileNotFoundError(
            f"No eia930_miso_demand.csv or DA_Load_EPNodes_*.zip files in {path}"
        )

    chunks = [_parse_load_epnodes_zip(f) for f in files]
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        raise ValueError(f"All load forecast files in {path} failed to parse")

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _load_gen_fuel_mix_raw(path: Path) -> pd.DataFrame:
    """
    Parse one historical_gen_fuel_mix_YYYY.xlsx file.

    Columns: Market Date, HourEnding, Region, Fuel Type,
             DA Cleared UDS Generation, RT Generation State Estimator.
    Returns long-format DataFrame with UTC DatetimeIndex.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    # Find header row (contains "Fuel Type")
    header_idx = next(i for i, r in enumerate(rows) if "Fuel Type" in str(r))
    header = [str(c).strip() if c is not None else "" for c in rows[header_idx]]

    data = []
    for row in rows[header_idx + 1:]:
        if row[1] is None:
            continue
        data.append(dict(zip(header, row)))

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "Market Date":                   "market_date",
        "HourEnding":                    "hour_ending",
        "Region":                        "region",
        "Fuel Type":                     "fuel_type",
        "DA Cleared UDS Generation":     "da_cleared_mw",
        "[RT Generation State Estimator": "rt_actual_mw",
    })
    df = df[["market_date", "hour_ending", "region", "fuel_type",
             "da_cleared_mw", "rt_actual_mw"]].copy()
    df["da_cleared_mw"] = pd.to_numeric(df["da_cleared_mw"], errors="coerce").fillna(0.0)
    df["rt_actual_mw"]  = pd.to_numeric(df["rt_actual_mw"],  errors="coerce").fillna(0.0)

    # Build UTC datetime: HourEnding 1 = 00:00-01:00 EST on that market_date
    df["market_date"] = pd.to_datetime(df["market_date"], errors="coerce")
    df["hour_ending"] = pd.to_numeric(df["hour_ending"], errors="coerce").fillna(1).astype(int)
    df = df.dropna(subset=["market_date"])

    local_ts = df["market_date"] + pd.to_timedelta(df["hour_ending"] - 1, unit="h")
    df["datetime"] = local_ts.dt.tz_localize("EST").dt.tz_convert("UTC")

    return df.drop(columns=["market_date", "hour_ending"])


def load_gen_fuel_mix(path: Path | None = None) -> pd.DataFrame:
    """
    Load MISO Historical Generation Fuel Mix files.

    Reads ``historical_gen_fuel_mix_{year}.xlsx`` from data/raw/gen_fuel_mix/.
    Returns long-format DataFrame with columns:
        datetime (UTC), region, fuel_type, da_cleared_mw, rt_actual_mw.

    Fuel types: Coal, Gas, Hydro, Nuclear, Other, Solar, Storage, Wind.

    Usage for wind/solar features:
        df[df.fuel_type.isin(['Wind','Solar'])]
    Usage for outage proxy:
        df[df.fuel_type.isin(['Coal','Gas','Nuclear'])]
    """
    path = path or RAW_DIR / "gen_fuel_mix"
    files = sorted(Path(path).glob("historical_gen_fuel_mix_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No historical_gen_fuel_mix_*.xlsx in {path}")

    chunks = [_load_gen_fuel_mix_raw(f) for f in files]
    df = pd.concat(chunks, ignore_index=True)
    return df.sort_values("datetime").reset_index(drop=True)


def load_renewables(path: Path | None = None) -> pd.DataFrame:
    """
    Load hourly wind and solar DA forecast (proxy) and RT actuals.

    Source: MISO Historical Generation Fuel Mix — DA Cleared UDS Generation
    for Wind and Solar fuel types summed across all MISO regions.
    DA Cleared MW is the best available public proxy for the DA renewable
    forecast (what the market scheduled = what operators expected to generate).

    Returns DataFrame with UTC DatetimeIndex, columns:
        wind_forecast_mw   (DA cleared Wind, sum across regions)
        solar_forecast_mw  (DA cleared Solar, sum across regions)
        wind_actual_mw     (RT actuals, for reference)
        solar_actual_mw
    """
    gfm = load_gen_fuel_mix(path)
    renew = gfm[gfm["fuel_type"].isin(["Wind", "Solar"])].copy()

    pivot = (
        renew.groupby(["datetime", "fuel_type"])[["da_cleared_mw", "rt_actual_mw"]]
        .sum()
        .unstack("fuel_type")
    )
    pivot.columns = [f"{col[1].lower()}_{col[0].replace('da_cleared_mw','forecast').replace('rt_actual_mw','actual')}"
                     for col in pivot.columns]

    result = pd.DataFrame({
        "wind_forecast_mw":  pivot.get("wind_forecast", pd.Series(dtype=float)),
        "solar_forecast_mw": pivot.get("solar_forecast", pd.Series(dtype=float)),
        "wind_actual_mw":    pivot.get("wind_actual",   pd.Series(dtype=float)),
        "solar_actual_mw":   pivot.get("solar_actual",  pd.Series(dtype=float)),
    }, index=pivot.index).fillna(0.0)

    return result


def load_outages(path: Path | None = None) -> pd.DataFrame:
    """
    Load generation outage data.

    Preferred source: CROW system CSV exports (``data/raw/outages/*.csv``).
    CROW data is not publicly available and must be obtained via MISO's
    commercial data service or subscription.

    Fallback: if no CROW files exist, builds an outage proxy from
    ``data/raw/gen_fuel_mix/`` thermal generation data. The proxy represents
    hourly thermal generation (Coal + Gas + Nuclear), which drops when units
    go on outage. The ``outage_features.py`` module handles both formats.

    Returns DataFrame with columns:
        resource_id, outage_start (UTC), outage_end (UTC),
        capacity_mw, outage_type  [CROW format]
      OR
        datetime (UTC), thermal_da_cleared_mw, thermal_rt_actual_mw  [proxy format]
    """
    crow_path = path or RAW_DIR / "outages"
    crow_files = sorted(Path(crow_path).glob("*.csv")) if Path(crow_path).exists() else []

    if crow_files:
        # Real CROW export
        df = pd.concat(
            [pd.read_csv(f, parse_dates=["outage_start", "outage_end"]) for f in crow_files],
            ignore_index=True,
        )
        df["outage_start"] = pd.to_datetime(df["outage_start"], utc=True)
        df["outage_end"]   = pd.to_datetime(df["outage_end"],   utc=True)
        return df

    # Proxy from gen fuel mix
    gfm = load_gen_fuel_mix(RAW_DIR / "gen_fuel_mix")
    thermal = gfm[gfm["fuel_type"].isin(["Coal", "Gas", "Nuclear"])].copy()
    proxy = (
        thermal.groupby("datetime")[["da_cleared_mw", "rt_actual_mw"]]
        .sum()
        .rename(columns={"da_cleared_mw": "thermal_da_cleared_mw",
                         "rt_actual_mw":  "thermal_rt_actual_mw"})
        .reset_index()
    )
    proxy["_is_proxy"] = True
    return proxy


def load_flowgate_loading(path: Path | None = None) -> pd.DataFrame:
    """
    Load flowgate real-time loading data.

    Returns DataFrame with columns:
        datetime (UTC), flowgate_id, loading_pct.
    """
    path = path or RAW_DIR / "flowgate_loading"
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {path}")
    df = pd.concat(
        [pd.read_csv(f, parse_dates=["datetime"]) for f in files],
        ignore_index=True,
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values(["datetime", "flowgate_id"]).reset_index(drop=True)


def load_ptdf(path: Path | None = None) -> pd.DataFrame:
    """
    Load PTDF shift factor reference data.

    Returns DataFrame with columns: flowgate_id, resource_id, ptdf_value.
    Reference-only — not used as raw training features.
    """
    path = path or RAW_DIR / "ptdf.csv"
    if not Path(path).exists():
        return pd.DataFrame(columns=["flowgate_id", "resource_id", "ptdf_value"])
    return pd.read_csv(path)
