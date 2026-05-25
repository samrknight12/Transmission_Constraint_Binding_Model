"""
Layer 1 feature pipeline — MISO transmission constraint binding model.

Assembles five feature groups into a single DataFrame with UTC DatetimeIndex:
  1. Temporal         — hour, DOW, month, peak/holiday indicators (cyclic encoded)
  2. Load forecast    — total MISO load, ramp rates, deviation from seasonal avg
  3. Outage           — system outage MW, PTDF-weighted flow impact, forced fraction
  4. Renewable        — wind/solar forecast, penetration %, expected ramps
  5. Flowgate         — loading %, trajectory, historical binding frequency

Target column (``binding``) is appended last and excluded from features at
train time. It is defined as |shadow_price| > 0.01 $/MWh.

Domain invariants enforced here:
  - All timestamps UTC; no timezone conversion beyond what sub-modules require.
  - Flowgate loading and shadow prices are filtered to the requested flowgate_id
    before any feature computation.
  - PTDF shift factors are consumed by outage_features for weighting only; they
    are never exposed as raw model inputs (reference data, not training data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .flowgate_features import build_flowgate_features
from .load_features import build_load_features
from .outage_features import build_outage_features
from .renewable_features import build_renewable_features
from .temporal import build_temporal_features

TARGET_COL = "binding"
_SHADOW_THRESHOLD = 0.01  # $/MWh — matches domain rule in CLAUDE.md


def build_target(shadow_price: pd.Series) -> pd.Series:
    """
    Binary binding label.

    Returns 1 where |shadow_price| > 0.01 $/MWh, 0 otherwise.
    Index must be a UTC DatetimeIndex.
    """
    return (shadow_price.abs() > _SHADOW_THRESHOLD).astype(np.int8).rename(TARGET_COL)


def build_layer1_features(
    flowgate_id: str,
    shadow_price_df: pd.DataFrame,
    load_forecast_df: pd.DataFrame,
    outage_df: pd.DataFrame,
    renewable_df: pd.DataFrame,
    flowgate_loading_df: pd.DataFrame,
    ptdf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the full Layer 1 feature matrix for one flowgate.

    The returned DataFrame has a UTC DatetimeIndex sorted ascending. The
    ``binding`` target column is the last column.  Callers should drop it
    before passing ``X`` to a model:

        X = features.drop(columns=["binding"])
        y = features["binding"]

    Parameters
    ----------
    flowgate_id : str
        MISO flowgate identifier, e.g. ``"LAKEFIELD.LAKFIELD 345 KV"``.
    shadow_price_df : DataFrame
        Columns: ``datetime`` (UTC), ``flowgate_id``, ``shadow_price``.
        May contain multiple flowgates — filtered internally.
    load_forecast_df : DataFrame
        DatetimeIndex (UTC), column ``load_forecast_mw`` (total MISO load).
    outage_df : DataFrame
        CROW export. Columns: ``resource_id``, ``outage_start`` (UTC),
        ``outage_end`` (UTC), ``capacity_mw``, ``outage_type``.
    renewable_df : DataFrame
        DatetimeIndex (UTC). Columns: ``wind_forecast_mw``, ``solar_forecast_mw``.
    flowgate_loading_df : DataFrame
        Columns: ``datetime`` (UTC), ``flowgate_id``, ``loading_pct``.
        May contain multiple flowgates — filtered internally.
    ptdf_df : DataFrame
        Reference data. Columns: ``flowgate_id``, ``resource_id``, ``ptdf_value``.
        Values are used as weights for outage impact only — never as raw features.

    Returns
    -------
    pd.DataFrame
        Feature matrix with UTC DatetimeIndex. Shape: (n_hours, n_features + 1).
        No NaN values — sub-modules fill missing data with 0 or a domain-appropriate
        sentinel before returning.
    """
    # ── Build complete hourly UTC index across the full data period ──────────
    # load_forecast_df covers all hours (not sparse like shadow_price_df).
    # shadow_price_df only contains binding events — non-binding hours are absent.
    # Reindexing fills missing hours with shadow_price=0 → binding label=0.
    idx = load_forecast_df.index.sort_values()

    # groupby deduplicates hours where the same flowgate appears under multiple
    # contingencies; max |shadow_price| preserves the binding signal correctly.
    fg_shadow_sparse = (
        shadow_price_df[shadow_price_df["flowgate_id"] == flowgate_id]
        .set_index("datetime")["shadow_price"]
        .sort_index()
        .groupby(level=0)
        .agg(lambda x: x.abs().max())
    )
    shadow_complete = fg_shadow_sparse.reindex(idx, fill_value=0.0)
    target = build_target(shadow_complete)

    fg_loading = (
        flowgate_loading_df[flowgate_loading_df["flowgate_id"] == flowgate_id]
        .set_index("datetime")
        .sort_index()
    )

    # ── Build feature groups (each returns DataFrame aligned to idx) ──────────
    # Attach load_forecast_mw to renewable_df so penetration % can be computed.
    renew_with_load = renewable_df.join(load_forecast_df[["load_forecast_mw"]], how="left")

    frames: list[pd.DataFrame] = [
        build_temporal_features(idx),
        build_load_features(load_forecast_df, idx),
        build_outage_features(outage_df, ptdf_df, flowgate_id, idx),
        build_renewable_features(renew_with_load, idx),
        build_flowgate_features(fg_loading, target, idx),
    ]

    features = pd.concat(frames, axis=1).reindex(idx)
    features[TARGET_COL] = target
    return features
