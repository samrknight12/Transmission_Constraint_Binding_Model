"""
Generation outage features weighted by PTDF shift factors.

Two data modes, detected automatically from the outage_df schema:

  CROW mode   — outage_df has columns: resource_id, outage_start, outage_end,
                capacity_mw, outage_type.  Produces PTDF-weighted MW features.
                Requires MISO CROW system data (commercial/subscription source).

  Proxy mode  — outage_df has column _is_proxy=True and columns:
                datetime, thermal_da_cleared_mw, thermal_rt_actual_mw.
                Source: MISO Historical Generation Fuel Mix (public).
                Produces thermal generation level and deviation features.
                PTDF-weighted impact is set to 0 (resource-level data unavailable).

PTDF values are used ONLY as weights in CROW mode — never as raw features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _expand_outages_to_hourly(
    outage_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Expand per-outage records (start/end/MW) to per-hour totals.

    Uses numpy vectorised mask indexing — O(n_outages × n_hours) but avoids
    Python-level row iteration on the index array.

    Expected outage_df columns:
        resource_id, outage_start (UTC), outage_end (UTC),
        capacity_mw, outage_type ('planned' | 'forced')
    """
    n = len(target_index)
    idx_ns = target_index.asi8  # int64 nanoseconds — fast range check

    total   = np.zeros(n, dtype=np.float64)
    planned = np.zeros(n, dtype=np.float64)
    forced  = np.zeros(n, dtype=np.float64)
    count   = np.zeros(n, dtype=np.int32)

    if outage_df.empty:
        return pd.DataFrame(
            {"outage_mw_total": total, "planned_outage_mw": planned,
             "forced_outage_mw": forced, "outage_count": count},
            index=target_index,
        )

    starts_ns = pd.to_datetime(outage_df["outage_start"]).values.astype(np.int64)
    ends_ns   = pd.to_datetime(outage_df["outage_end"]).values.astype(np.int64)
    caps      = outage_df["capacity_mw"].values.astype(np.float64)

    if "outage_type" in outage_df.columns:
        is_forced = (outage_df["outage_type"].str.lower() == "forced").values
    else:
        is_forced = np.zeros(len(outage_df), dtype=bool)

    for i in range(len(outage_df)):
        mask = (idx_ns >= starts_ns[i]) & (idx_ns < ends_ns[i])
        total[mask]  += caps[i]
        count[mask]  += 1
        if is_forced[i]:
            forced[mask]  += caps[i]
        else:
            planned[mask] += caps[i]

    return pd.DataFrame(
        {"outage_mw_total": total, "planned_outage_mw": planned,
         "forced_outage_mw": forced, "outage_count": count},
        index=target_index,
    )


def _build_proxy_outage_features(
    outage_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build outage proxy features from Historical Generation Fuel Mix thermal data.

    Features:
      - thermal_da_cleared_mw      : total Coal+Gas+Nuclear DA cleared generation
      - thermal_rt_actual_mw       : total Coal+Gas+Nuclear RT actual generation
      - thermal_deviation_30d_mw   : deviation from 30-day rolling average (outage signal)
      - thermal_change_24h_mw      : 24h change (ramp-down = potential forced outage)
      - thermal_rt_vs_da_gap_mw    : RT - DA (negative = less generation than expected)
    """
    proxy = outage_df.set_index("datetime").reindex(target_index)

    da  = proxy["thermal_da_cleared_mw"].fillna(method="ffill").fillna(0.0)
    rt  = proxy["thermal_rt_actual_mw"].fillna(method="ffill").fillna(0.0)

    rolling_avg_30d     = da.rolling(30 * 24, min_periods=24 * 7).mean()
    thermal_deviation   = (da - rolling_avg_30d).fillna(0.0)
    thermal_change_24h  = da.diff(24).fillna(0.0)
    rt_vs_da_gap        = (rt - da).fillna(0.0)

    return pd.DataFrame(
        {
            "outage_mw_total":         da,       # alias for CROW-mode compat
            "planned_outage_mw":       da * 0,   # unknown split in proxy mode
            "forced_outage_mw":        da * 0,
            "outage_count":            da * 0,
            "outage_mw_ptdf_weighted": da * 0,   # resource-level data unavailable
            "outage_mw_change_24h":    thermal_change_24h,
            "forced_outage_fraction":  da * 0,
            # Proxy-specific features
            "thermal_da_cleared_mw":   da,
            "thermal_rt_actual_mw":    rt,
            "thermal_deviation_30d_mw": thermal_deviation,
            "thermal_rt_vs_da_gap_mw": rt_vs_da_gap,
        },
        index=target_index,
    )


def build_outage_features(
    outage_df: pd.DataFrame,
    ptdf_df: pd.DataFrame,
    flowgate_id: str,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build outage feature columns for a single flowgate.

    Detects data mode from outage_df schema:
      - CROW mode: has 'outage_start' column → resource-level PTDF-weighted features
      - Proxy mode: has '_is_proxy' column  → thermal generation deviation features

    Parameters
    ----------
    outage_df : DataFrame
        Either CROW export or gen_fuel_mix proxy (see module docstring).
    ptdf_df : DataFrame
        Reference-only. Columns: [flowgate_id, resource_id, ptdf_value].
        Raw PTDF values NOT included in returned feature matrix.
    flowgate_id : str
        Target flowgate identifier.
    target_index : DatetimeIndex (UTC)
        Timestamps to align output to.

    Returns
    -------
    DataFrame aligned to target_index. Column set differs slightly by mode
    but always includes outage_mw_total and outage_mw_ptdf_weighted for
    downstream compatibility.
    """
    # ── Proxy mode (gen_fuel_mix thermal data) ────────────────────────────────
    if "_is_proxy" in outage_df.columns:
        return _build_proxy_outage_features(outage_df, target_index)

    # ── CROW mode ─────────────────────────────────────────────────────────────
    hourly = _expand_outages_to_hourly(outage_df, target_index)

    fg_ptdf = (
        ptdf_df[ptdf_df["flowgate_id"] == flowgate_id]
        .set_index("resource_id")["ptdf_value"]
    )
    ptdf_weighted = np.zeros(len(target_index), dtype=np.float64)

    if not fg_ptdf.empty:
        idx_ns = target_index.asi8
        for resource_id, ptdf_val in fg_ptdf.items():
            res_outages = outage_df[outage_df["resource_id"] == resource_id]
            if res_outages.empty:
                continue
            starts_ns = pd.to_datetime(res_outages["outage_start"]).values.astype(np.int64)
            ends_ns   = pd.to_datetime(res_outages["outage_end"]).values.astype(np.int64)
            caps      = res_outages["capacity_mw"].values.astype(np.float64)
            for i in range(len(res_outages)):
                mask = (idx_ns >= starts_ns[i]) & (idx_ns < ends_ns[i])
                ptdf_weighted[mask] += caps[i] * abs(ptdf_val)

    hourly["outage_mw_ptdf_weighted"] = ptdf_weighted
    hourly["outage_mw_change_24h"] = hourly["outage_mw_total"].diff(24).fillna(0.0)
    hourly["forced_outage_fraction"] = (
        hourly["forced_outage_mw"]
        / hourly["outage_mw_total"].replace(0.0, np.nan)
    ).fillna(0.0)

    return hourly
