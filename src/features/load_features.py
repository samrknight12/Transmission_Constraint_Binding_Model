"""
MISO DA load forecast features.

Requires a pre-aggregated total MISO load forecast series (sum across zones).
All windows are backwards-looking — no forward leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Rolling window sizes (hourly data assumed)
_WINDOW_30D = 30 * 24
_WINDOW_365D = 365 * 24


def build_load_features(
    load_forecast_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    load_col: str = "load_forecast_mw",
) -> pd.DataFrame:
    """
    Build load forecast feature columns.

    Parameters
    ----------
    load_forecast_df : DataFrame
        DatetimeIndex (UTC), must contain `load_col` (total MISO load forecast MW).
        Aggregated across all zones before passing in.
    target_index : DatetimeIndex (UTC)
        Timestamps to align output to.
    load_col : str
        Column name for total MISO load forecast MW.

    Returns
    -------
    DataFrame aligned to target_index.
    """
    load = load_forecast_df[load_col].reindex(target_index)

    # Rolling seasonal peak — normalises load to fraction of recent peak
    rolling_peak = load.rolling(window=_WINDOW_365D, min_periods=_WINDOW_30D).max()
    load_pct_of_peak = (load / rolling_peak.replace(0.0, np.nan)).fillna(0.0)

    # Deviation from rolling 30-day mean — captures anomalous load days
    rolling_avg_30d = load.rolling(window=_WINDOW_30D, min_periods=24 * 7).mean()
    load_deviation_mw = (load - rolling_avg_30d).fillna(0.0)

    # Backward-looking ramp rates
    load_change_1h  = load.diff(1).fillna(0.0)
    load_change_4h  = load.diff(4).fillna(0.0)
    load_change_24h = load.diff(24).fillna(0.0)

    # Forward-looking forecast horizon (legitimate: these come from the forecast file,
    # not from future actuals — i.e. the DA forecast already covers these hours)
    load_ahead_1h = (load.shift(-1) - load).fillna(0.0)
    load_ahead_4h = (load.shift(-4) - load).fillna(0.0)

    return pd.DataFrame(
        {
            "load_forecast_mw":           load.fillna(0.0),
            "load_pct_of_peak":           load_pct_of_peak,
            "load_deviation_from_avg_mw": load_deviation_mw,
            "load_change_1h_mw":          load_change_1h,
            "load_change_4h_mw":          load_change_4h,
            "load_change_24h_mw":         load_change_24h,
            "load_forecast_ahead_1h_mw":  load_ahead_1h,
            "load_forecast_ahead_4h_mw":  load_ahead_4h,
        },
        index=target_index,
    )
