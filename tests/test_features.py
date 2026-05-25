"""
Tests for Layer 1 feature pipeline.

Focus areas:
  - Correct output shape and index alignment
  - No NaN values after filling
  - Cyclic feature bounds
  - Peak-hour logic (weekday-only, not holiday)
  - Leakage guard: binding frequency features use only past data
  - Outage presence/absence in correct time windows
  - Target definition (|shadow_price| > 0.01)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TARGET_COL, build_layer1_features, build_target
from src.features.flowgate_features import build_flowgate_features
from src.features.load_features import build_load_features
from src.features.outage_features import build_outage_features
from src.features.renewable_features import build_renewable_features
from src.features.temporal import build_temporal_features

FLOWGATE_ID = "TEST.GATE 345 KV"


# ── Temporal ──────────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_shape_and_index(self, utc_index):
        df = build_temporal_features(utc_index)
        assert len(df) == len(utc_index)
        assert df.index.equals(utc_index)

    def test_cyclic_feature_bounds(self, utc_index):
        df = build_temporal_features(utc_index)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert df[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"

    def test_peak_hour_weekday_only(self, utc_index):
        df = build_temporal_features(utc_index)
        weekend_mask = df["is_weekend"] == 1
        assert (df.loc[weekend_mask, "is_peak_hour"] == 0).all()

    def test_no_nulls(self, utc_index):
        df = build_temporal_features(utc_index)
        assert df.isnull().sum().sum() == 0

    def test_season_values(self, utc_index):
        df = build_temporal_features(utc_index)
        assert df["season"].isin([1, 2, 3, 4]).all()


# ── Load ──────────────────────────────────────────────────────────────────────

class TestLoadFeatures:
    def test_shape(self, utc_index, load_forecast_df):
        df = build_load_features(load_forecast_df, utc_index)
        assert len(df) == len(utc_index)

    def test_no_nulls(self, utc_index, load_forecast_df):
        df = build_load_features(load_forecast_df, utc_index)
        assert df.isnull().sum().sum() == 0

    def test_pct_of_peak_bounds(self, utc_index, load_forecast_df):
        df = build_load_features(load_forecast_df, utc_index)
        assert (df["load_pct_of_peak"] >= 0.0).all()
        assert (df["load_pct_of_peak"] <= 1.0).all()


# ── Renewables ────────────────────────────────────────────────────────────────

class TestRenewableFeatures:
    def test_shape(self, utc_index, renewable_df):
        df = build_renewable_features(renewable_df, utc_index)
        assert len(df) == len(utc_index)

    def test_total_equals_sum(self, utc_index, renewable_df):
        df = build_renewable_features(renewable_df, utc_index)
        expected = (
            renewable_df["wind_forecast_mw"] + renewable_df["solar_forecast_mw"]
        ).reindex(utc_index)
        pd.testing.assert_series_equal(
            df["renewable_total_mw"], expected, check_names=False
        )

    def test_no_nulls_in_core_cols(self, utc_index, renewable_df):
        df = build_renewable_features(renewable_df, utc_index)
        core = ["wind_forecast_mw", "solar_forecast_mw", "renewable_total_mw"]
        assert df[core].isnull().sum().sum() == 0


# ── Outage ────────────────────────────────────────────────────────────────────

class TestOutageFeatures:
    def test_shape(self, utc_index, outage_df, ptdf_df):
        df = build_outage_features(outage_df, ptdf_df, FLOWGATE_ID, utc_index)
        assert len(df) == len(utc_index)

    def test_zero_before_outage_window(self, utc_index, outage_df, ptdf_df):
        df = build_outage_features(outage_df, ptdf_df, FLOWGATE_ID, utc_index)
        # outage starts at index[4]; index[0:4] should have zero total outage
        assert (df.iloc[:4]["outage_mw_total"] == 0.0).all()

    def test_nonzero_during_outage_window(self, utc_index, outage_df, ptdf_df):
        df = build_outage_features(outage_df, ptdf_df, FLOWGATE_ID, utc_index)
        assert df.iloc[5]["outage_mw_total"] > 0.0

    def test_ptdf_weighted_non_negative(self, utc_index, outage_df, ptdf_df):
        df = build_outage_features(outage_df, ptdf_df, FLOWGATE_ID, utc_index)
        assert (df["outage_mw_ptdf_weighted"] >= 0.0).all()

    def test_forced_fraction_in_01(self, utc_index, outage_df, ptdf_df):
        df = build_outage_features(outage_df, ptdf_df, FLOWGATE_ID, utc_index)
        assert df["forced_outage_fraction"].between(0.0, 1.0).all()


# ── Flowgate ──────────────────────────────────────────────────────────────────

class TestFlowgateFeatures:
    def _make_loading(self, utc_index, flowgate_loading_df):
        return flowgate_loading_df.drop(columns=["flowgate_id"])

    def _make_target(self, shadow_price_df):
        return build_target(shadow_price_df.set_index("datetime")["shadow_price"])

    def test_shape(self, utc_index, flowgate_loading_df, shadow_price_df):
        target  = self._make_target(shadow_price_df)
        loading = self._make_loading(utc_index, flowgate_loading_df)
        df = build_flowgate_features(loading, target, utc_index)
        assert len(df) == len(utc_index)

    def test_distance_to_limit_non_negative(self, utc_index, flowgate_loading_df, shadow_price_df):
        target  = self._make_target(shadow_price_df)
        loading = self._make_loading(utc_index, flowgate_loading_df)
        df = build_flowgate_features(loading, target, utc_index)
        assert (df["flowgate_distance_to_limit"] >= 0.0).all()

    def test_binding_freq_no_future_leakage(self, utc_index, flowgate_loading_df, shadow_price_df):
        """First position must have binding_freq = 0 (no history before t=0)."""
        target  = self._make_target(shadow_price_df)
        loading = self._make_loading(utc_index, flowgate_loading_df)
        df = build_flowgate_features(loading, target, utc_index)
        # At t=0 there is no prior history — rolling over shifted series → NaN → filled 0
        assert df.iloc[0]["flowgate_binding_freq_7d"] == 0.0

    def test_hours_since_binding_non_negative(self, utc_index, flowgate_loading_df, shadow_price_df):
        target  = self._make_target(shadow_price_df)
        loading = self._make_loading(utc_index, flowgate_loading_df)
        df = build_flowgate_features(loading, target, utc_index)
        assert (df["flowgate_hours_since_binding"] >= 0.0).all()


# ── End-to-end ────────────────────────────────────────────────────────────────

class TestBuildTarget:
    def test_binary_values(self, shadow_price_df):
        target = build_target(shadow_price_df.set_index("datetime")["shadow_price"])
        assert set(target.unique()).issubset({0, 1})

    def test_threshold_boundary(self):
        prices = pd.Series([0.0, 0.009, 0.01, 0.011, 1.0, -0.5, -0.011])
        result = build_target(prices)
        expected = pd.Series([0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_negative_prices_bind(self):
        prices = pd.Series([-5.0, -0.02, 0.0, 0.02])
        result = build_target(prices)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 1
        assert result.iloc[2] == 0
        assert result.iloc[3] == 1


class TestBuildLayer1Features:
    def test_output_has_target_last(
        self, utc_index, shadow_price_df, load_forecast_df,
        outage_df, ptdf_df, renewable_df, flowgate_loading_df
    ):
        # flowgate_loading_df needs datetime column, not index
        loading = flowgate_loading_df.reset_index().rename(columns={"index": "datetime"})
        features = build_layer1_features(
            FLOWGATE_ID,
            shadow_price_df,
            load_forecast_df,
            outage_df,
            renewable_df,
            loading,
            ptdf_df,
        )
        assert features.columns[-1] == TARGET_COL

    def test_no_raw_ptdf_in_features(
        self, utc_index, shadow_price_df, load_forecast_df,
        outage_df, ptdf_df, renewable_df, flowgate_loading_df
    ):
        loading = flowgate_loading_df.reset_index().rename(columns={"index": "datetime"})
        features = build_layer1_features(
            FLOWGATE_ID, shadow_price_df, load_forecast_df,
            outage_df, renewable_df, loading, ptdf_df,
        )
        # Raw PTDF columns must never appear in the feature matrix
        assert not any("ptdf_value" in col for col in features.columns)
