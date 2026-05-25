"""
Wind and solar forecast features for MISO constraint binding model.

High renewable penetration raises congestion risk on specific corridors
(e.g. Iowa/Minnesota wind export paths). Ramp features capture expected
variability that stresses transmission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_renewable_features(
    renewable_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    wind_col: str = "wind_forecast_mw",
    solar_col: str = "solar_forecast_mw",
    load_col: str | None = "load_forecast_mw",
) -> pd.DataFrame:
    """
    Build wind and solar forecast feature columns.

    Parameters
    ----------
    renewable_df : DataFrame
        DatetimeIndex (UTC). Required columns: wind_forecast_mw, solar_forecast_mw.
        Optional column: load_forecast_mw (enables penetration % feature).
    target_index : DatetimeIndex (UTC)
        Timestamps to align output to.

    Returns
    -------
    DataFrame aligned to target_index.
    """
    wind  = renewable_df[wind_col].reindex(target_index).fillna(0.0)
    solar = renewable_df[solar_col].reindex(target_index).fillna(0.0)
    total = wind + solar

    # Backward-looking ramps (actuals / prior forecasts available at dispatch time)
    wind_ramp_1h  = wind.diff(1).fillna(0.0)
    wind_ramp_4h  = wind.diff(4).fillna(0.0)
    solar_ramp_1h = solar.diff(1).fillna(0.0)

    # Forward-looking DA forecast ramps (from the same DA forecast file — not leakage)
    wind_ahead_1h = (wind.shift(-1) - wind).fillna(0.0)
    wind_ahead_4h = (wind.shift(-4) - wind).fillna(0.0)

    # Rolling 4h forecast variability (std) — proxy for how "uncertain" wind is
    wind_var_4h = wind.rolling(4, min_periods=2).std().fillna(0.0)

    # Renewable penetration requires load in the same DataFrame
    if load_col and load_col in renewable_df.columns:
        load = (
            renewable_df[load_col]
            .reindex(target_index)
            .ffill()   # fill short gaps (e.g. EIA-930 data outages)
            .bfill()
        )
        renewable_penetration = (total / load.replace(0.0, np.nan) * 100.0).fillna(0.0)
    else:
        renewable_penetration = pd.Series(0.0, index=target_index, name="renewable_penetration_pct")

    return pd.DataFrame(
        {
            "wind_forecast_mw":          wind,
            "solar_forecast_mw":         solar,
            "renewable_total_mw":        total,
            "renewable_penetration_pct": renewable_penetration,
            "wind_ramp_1h_mw":           wind_ramp_1h,
            "wind_ramp_4h_mw":           wind_ramp_4h,
            "solar_ramp_1h_mw":          solar_ramp_1h,
            "wind_ahead_1h_mw":          wind_ahead_1h,
            "wind_ahead_4h_mw":          wind_ahead_4h,
            "wind_variability_4h_mw":    wind_var_4h,
        },
        index=target_index,
    )
