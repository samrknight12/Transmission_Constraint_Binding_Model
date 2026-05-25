"""
Master dataset builder for MISO transmission constraint binding model.

Assembles the full feature matrix for all target flowgates (binding_rate >= 3%)
and saves per-flowgate parquet files plus a stacked master parquet.

Usage:
    python src/features/build_master_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root when run as a script
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loaders import (
    load_binding_constraints,
    load_flowgate_loading,
    load_load_forecasts,
    load_outages,
    load_ptdf,
    load_renewables,
)
from src.features.build_features import build_layer1_features

_BINDING_RATE_MIN = 0.03
_OUT_DIR = Path("data/processed")
_FEATURES_DIR = _OUT_DIR / "features"

# Sentinel returned by flowgate_features._hours_since_last_binding when no prior event exists
_HOURS_SENTINEL = 8_760.0


# ── Flowgate quality tier ─────────────────────────────────────────────────────

def assign_flowgate_tier(binding_rate: float, observed_loading_pct: float) -> str:
    """
    Classify a flowgate by the reliability of its loading features.

    "synthetic_only"  observed_loading_pct == 0   — all loading_pct values are
                      the 85.0 fill; no RT confirmation exists at any hour.
    "low_signal"      0 < observed_loading_pct < 3% — sparse RT confirmation;
                      loading features are mostly synthetic.
    "high_signal"     observed_loading_pct >= 3%  — enough RT binding events
                      to treat loading_pct as a reliable feature.
    """
    if observed_loading_pct == 0:
        return "synthetic_only"
    if observed_loading_pct < 0.03:
        return "low_signal"
    return "high_signal"


# ── Step 2 helper ────────────────────────────────────────────────────────────

def _target_flowgates(bc_df: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute per-flowgate binding rate and return those at or above the threshold.

    Returns a DataFrame indexed by flowgate_id with column binding_rate,
    sorted descending.
    """
    n_hours = len(hourly_index)
    binding_events = bc_df[bc_df["shadow_price"].abs() > 0.01]
    counts = (
        binding_events.groupby("flowgate_id")["datetime"]
        .nunique()
        .rename("binding_hours")
    )
    rates = (counts / n_hours).rename("binding_rate").to_frame()
    return rates[rates["binding_rate"] >= _BINDING_RATE_MIN].sort_values(
        "binding_rate", ascending=False
    )


# ── Step 3 supplement ────────────────────────────────────────────────────────

def _add_supplemental_features(
    features: pd.DataFrame,
    load_df: pd.DataFrame,
    outage_df: pd.DataFrame,
    renewable_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append features not produced by build_layer1_features.

    Added columns
    -------------
    load_deviation_from_7d_mean   : load vs. backward 7d rolling mean
    wind_forecast_rolling_mae_7d  : 7d rolling std of wind forecast (uncertainty proxy)
    thermal_outage_mw             : alias for outage_mw_total (proxy or CROW)
    outage_pct_of_capacity        : thermal_outage / 30d rolling-max thermal
    is_outage_proxy               : 1 if gen_fuel_mix proxy mode, 0 for CROW
    season                        : re-encoded 0-3 (temporal.py returns 1-4)
    """
    idx = features.index
    is_proxy = "_is_proxy" in outage_df.columns

    # load_deviation_from_7d_mean — strictly backward-looking via rolling mean
    load = load_df["load_forecast_mw"].reindex(idx)
    rolling_7d_mean = load.rolling(7 * 24, min_periods=24).mean()
    features["load_deviation_from_7d_mean"] = (load - rolling_7d_mean).fillna(0.0)

    # wind_forecast_rolling_mae_7d — rolling 7d std as forecast-uncertainty proxy
    # (DA wind forecasts have no concurrent actuals; std captures expected variability)
    wind = renewable_df["wind_forecast_mw"].reindex(idx).fillna(0.0)
    features["wind_forecast_rolling_mae_7d"] = (
        wind.rolling(7 * 24, min_periods=24).std().fillna(0.0)
    )

    # thermal_outage_mw — in proxy mode outage_mw_total holds thermal DA cleared MW;
    # in CROW mode it holds actual offline capacity MW
    features["thermal_outage_mw"] = features["outage_mw_total"].copy()

    # outage_pct_of_capacity — thermal level relative to 30d rolling max
    thermal = features["outage_mw_total"]
    rolling_max_30d = thermal.rolling(30 * 24, min_periods=7 * 24).max()
    features["outage_pct_of_capacity"] = (
        (thermal / rolling_max_30d.replace(0.0, np.nan)) * 100.0
    ).fillna(0.0)

    # is_outage_proxy flag
    features["is_outage_proxy"] = np.int8(is_proxy)

    # season: re-encode to 0-3 (temporal.py returns 1=winter … 4=fall)
    if "season" in features.columns:
        features["season"] = (features["season"] - 1).astype(np.int8)

    return features


# ── Step 4: leakage checks ───────────────────────────────────────────────────

def _leakage_check(features: pd.DataFrame, flowgate_id: str) -> None:
    """
    Assert no data leakage in rolling binding features and no spurious target correlation.

    Checks
    ------
    1. binding_freq_trailing_30d == 0 at the first-ever binding event, proving shift(1)
       is applied before the rolling window (no same-hour label included).
    2. hours_since_last_binding equals the sentinel (~8760) at the first-ever binding
       event, confirming no future binding timestamps are visible.
    3. No numeric feature has |corr| > 0.95 with the binary target, which would
       indicate that target information leaked into a feature column.
    """
    target = features["binding"]
    first_binding = target[target == 1]

    if not first_binding.empty:
        t0 = first_binding.index[0]

        freq_30d = features["flowgate_binding_freq_30d"]
        assert freq_30d.loc[t0] == 0.0, (
            f"[{flowgate_id}] Leakage: flowgate_binding_freq_30d={freq_30d.loc[t0]:.4f} "
            f"at first binding event — shift(1) may be missing"
        )

        hours_since = features["flowgate_hours_since_binding"]
        assert hours_since.loc[t0] >= _HOURS_SENTINEL * 0.99, (
            f"[{flowgate_id}] Leakage: flowgate_hours_since_binding={hours_since.loc[t0]:.1f} "
            f"at first binding event — expected sentinel {_HOURS_SENTINEL}"
        )

    # High-correlation gate
    skip = {"binding"}
    for col in features.columns:
        if col in skip:
            continue
        series = features[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if series.std(ddof=0) == 0:
            continue  # constant feature; corr undefined, not a leakage signal
        corr = series.corr(target.astype(float))
        if abs(corr) > 0.95:
            raise ValueError(
                f"[{flowgate_id}] Possible leakage: '{col}' has |corr|={abs(corr):.4f} "
                f"with target (threshold 0.95)"
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== MISO Master Dataset Builder ===\n")

    # ── Step 1: Load all sources ─────────────────────────────────────────────
    print("Loading data sources...")
    bc_df = load_binding_constraints()
    load_df = load_load_forecasts()
    renew_df = load_renewables()
    outage_df = load_outages()
    loading_df = load_flowgate_loading()
    ptdf_df = load_ptdf()

    mode = "proxy" if "_is_proxy" in outage_df.columns else "CROW"
    print(f"  Binding constraints : {len(bc_df):>10,} events")
    print(
        f"  Load forecast       : {len(load_df):>10,} hours  "
        f"({load_df.index.min()} -> {load_df.index.max()})"
    )
    print(f"  Renewables          : {len(renew_df):>10,} hours")
    print(f"  Outages             : {len(outage_df):>10,} rows  [{mode} mode]")
    print(
        f"  Flowgate loading    : {len(loading_df):>10,} rows  "
        f"({loading_df['flowgate_id'].nunique():,} unique flowgates)"
    )

    # ── Step 2: Identify target flowgates ────────────────────────────────────
    print(f"\nIdentifying target flowgates (binding_rate >= {_BINDING_RATE_MIN:.0%})...")
    target_fg = _target_flowgates(bc_df, load_df.index)
    # Compute observed_loading_pct: fraction of hours each flowgate appeared in RT data.
    # Matches flowgate_loading_pct_is_observed.mean() produced by build_flowgate_features.
    total_hours = len(load_df.index)
    observed_counts = loading_df.groupby("flowgate_id")["datetime"].nunique()
    target_fg["observed_loading_pct"] = (
        observed_counts / total_hours
    ).reindex(target_fg.index).fillna(0.0)
    target_fg["tier"] = [
        assign_flowgate_tier(row["binding_rate"], row["observed_loading_pct"])
        for _, row in target_fg.iterrows()
    ]

    print(f"  {len(target_fg)} flowgates qualify:\n")
    for fg_id, row in target_fg.iterrows():
        print(
            f"    {fg_id:<55s}  {row['binding_rate']:.2%}"
            f"  obs_loading={row['observed_loading_pct']:.2%}"
            f"  [{row['tier']}]"
        )

    tier_counts = target_fg["tier"].value_counts()
    print(f"\n  Tier distribution: " + "  ".join(
        f"{t}={n}" for t, n in tier_counts.items()
    ))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    target_fg.to_csv(_OUT_DIR / "target_flowgates.csv")
    print("  Saved -> data/processed/target_flowgates.csv")

    # ── Steps 3–5: Build per-flowgate feature matrices ───────────────────────
    all_frames: list[pd.DataFrame] = []

    for flowgate_id in target_fg.index:
        print(f"\nBuilding: {flowgate_id}")

        # Step 3: Full Layer 1 feature matrix via existing pipeline
        features = build_layer1_features(
            flowgate_id=flowgate_id,
            shadow_price_df=bc_df,
            load_forecast_df=load_df,
            outage_df=outage_df,
            renewable_df=renew_df,
            flowgate_loading_df=loading_df,
            ptdf_df=ptdf_df,
        )

        # Append supplemental features not in build_layer1_features
        features = _add_supplemental_features(features, load_df, outage_df, renew_df)

        # Prepend flowgate identifier so stacked master is self-describing
        features.insert(0, "flowgate_id", flowgate_id)

        # Step 4: Leakage check (pass without flowgate_id column)
        _leakage_check(features.drop(columns=["flowgate_id"]), flowgate_id)
        print("  Leakage check OK")

        n_total = len(features)
        n_binding = int(features["binding"].sum())
        print(
            f"  Shape: {features.shape}  |  "
            f"Binding: {n_binding:,}/{n_total:,} ({n_binding / n_total:.2%})  |  "
            f"{features.index.min()} -> {features.index.max()}"
        )

        # Step 5: Save per-flowgate parquet
        safe_id = (
            flowgate_id
            .replace("/", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )
        out_path = _FEATURES_DIR / f"{safe_id}.parquet"
        features.to_parquet(out_path)
        print(f"  Saved  -> {out_path}")

        all_frames.append(features)

    # ── Master stacked file ──────────────────────────────────────────────────
    print("\nStacking all flowgates into master dataset...")
    master = pd.concat(all_frames)
    master_path = _OUT_DIR / "master_dataset.parquet"
    master.to_parquet(master_path)

    print("\n=== Final Summary ===")
    print(f"Master shape    : {master.shape}")
    print(f"Date coverage   : {master.index.min()} -> {master.index.max()}")
    print(f"Flowgates       : {master['flowgate_id'].nunique()}")
    print(f"Overall binding : {master['binding'].mean():.2%}")
    print("\nPer-flowgate class balance:")
    for fg_id, grp in master.groupby("flowgate_id"):
        r = grp["binding"].mean()
        print(
            f"  {fg_id:<55s}  {r:.2%}  "
            f"({int(grp['binding'].sum()):,} / {len(grp):,})"
        )
    print(f"\nMaster dataset  -> {master_path}")


if __name__ == "__main__":
    main()
