"""
Flowgate loading and binding history features.

Critical leakage note:
  Rolling binding-frequency and hours-since-last-binding are computed on a
  SHIFTED target series (shift(1)) so position t uses only data from t-1 and
  earlier. Including t itself would leak the label into the feature.

PTDF shift factors are handled exclusively in outage_features.py.
Raw PTDF values are never included here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOW_7D  = 7  * 24
_WINDOW_30D = 30 * 24
_WINDOW_90D = 90 * 24
_DEFAULT_HOURS_NO_HISTORY = 8_760.0  # ~1 year sentinel when flowgate never bound


def _hours_since_last_binding(binding: pd.Series) -> pd.Series:
    """
    Compute hours elapsed since the most recent prior binding event.

    Uses shift(1) so the current timestep is excluded — no label leakage.
    Returns `_DEFAULT_HOURS_NO_HISTORY` when no prior binding event exists.
    """
    # Shift by 1: position t now holds the label from t-1
    past_binding = binding.shift(1, fill_value=0)

    # Forward-fill the timestamp of the last binding event
    binding_times = binding.index.to_series().where(past_binding == 1)
    last_binding_time = binding_times.ffill()

    hours_since = (
        (binding.index.to_series() - last_binding_time) / pd.Timedelta(hours=1)
    ).fillna(_DEFAULT_HOURS_NO_HISTORY)

    return hours_since.astype(np.float32)


def build_flowgate_features(
    loading_df: pd.DataFrame,
    binding_target: pd.Series,
    target_index: pd.DatetimeIndex,
    loading_pct_col: str = "loading_pct",
) -> pd.DataFrame:
    """
    Build flowgate loading and binding history features.

    Parameters
    ----------
    loading_df : DataFrame
        DatetimeIndex (UTC), must contain `loading_pct_col`.
        loading_pct = flow_mw / normal_limit_mw × 100.
    binding_target : Series
        DatetimeIndex (UTC), binary binding labels (1 = binding).
        Used ONLY for backwards-looking rolling statistics — no future leakage.
    target_index : DatetimeIndex (UTC)
        Timestamps to align output to.
    loading_pct_col : str
        Column name for loading percentage.

    Returns
    -------
    DataFrame aligned to target_index.
    """
    # 85.0 = synthetic baseline for non-binding hours (constraints that appear
    # in RT/DA binding data are typically loaded near their thermal limit).
    loading_raw = loading_df[loading_pct_col].reindex(target_index)
    loading_pct_is_observed = loading_raw.notna().astype(np.int8)
    loading = loading_raw.fillna(85.0)
    target  = binding_target.reindex(target_index).fillna(0).astype(np.int8)

    # Loading trajectory
    loading_chg_1h  = loading.diff(1).fillna(0.0)
    loading_chg_4h  = loading.diff(4).fillna(0.0)
    loading_chg_24h = loading.diff(24).fillna(0.0)
    distance_to_limit = (100.0 - loading).clip(lower=0.0)

    # Loading relative to recent maximum (30d rolling)
    rolling_max_30d = loading.rolling(_WINDOW_30D, min_periods=24).max()
    loading_pct_of_30d_max = (loading / rolling_max_30d.replace(0.0, np.nan)).fillna(0.0)

    # Binding frequency — shift(1) ensures strictly backward-looking (no leakage)
    past = target.shift(1, fill_value=0)
    binding_freq_7d  = past.rolling(_WINDOW_7D,  min_periods=24).mean().fillna(0.0)
    binding_freq_30d = past.rolling(_WINDOW_30D, min_periods=24 * 7).mean().fillna(0.0)
    binding_freq_90d = past.rolling(_WINDOW_90D, min_periods=24 * 14).mean().fillna(0.0)

    hours_since = _hours_since_last_binding(target)

    return pd.DataFrame(
        {
            "flowgate_loading_pct":           loading,
            "flowgate_loading_pct_is_observed": loading_pct_is_observed,
            "flowgate_loading_chg_1h":        loading_chg_1h,
            "flowgate_loading_chg_4h":        loading_chg_4h,
            "flowgate_loading_chg_24h":       loading_chg_24h,
            "flowgate_distance_to_limit":     distance_to_limit,
            "flowgate_pct_of_30d_max":        loading_pct_of_30d_max,
            "flowgate_binding_freq_7d":       binding_freq_7d,
            "flowgate_binding_freq_30d":      binding_freq_30d,
            "flowgate_binding_freq_90d":      binding_freq_90d,
            "flowgate_hours_since_binding":   hours_since,
        },
        index=target_index,
    )
