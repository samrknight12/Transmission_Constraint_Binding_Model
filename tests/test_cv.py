"""
Tests for MISOTimeSeriesSplit — 24h gap enforcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cv import MISOTimeSeriesSplit


@pytest.fixture
def hourly_df():
    idx = pd.date_range("2024-01-01", periods=60 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"x": rng.normal(size=len(idx)), TARGET_COL: rng.integers(0, 2, size=len(idx))},
        index=idx,
    )


TARGET_COL = "binding"


class TestMISOTimeSeriesSplit:
    def test_gap_at_least_24h(self, hourly_df):
        cv = MISOTimeSeriesSplit(n_splits=3, gap_hours=24)
        for train_df, val_df in cv.split_by_date(hourly_df):
            gap = val_df.index[0] - train_df.index[-1]
            assert gap >= pd.Timedelta(hours=24)

    def test_no_index_overlap(self, hourly_df):
        cv = MISOTimeSeriesSplit(n_splits=3, gap_hours=24)
        for train_df, val_df in cv.split_by_date(hourly_df):
            assert len(set(train_df.index) & set(val_df.index)) == 0

    def test_train_strictly_before_val(self, hourly_df):
        cv = MISOTimeSeriesSplit(n_splits=3, gap_hours=24)
        for train_df, val_df in cv.split_by_date(hourly_df):
            assert train_df.index.max() < val_df.index.min()

    def test_correct_n_splits(self, hourly_df):
        n = 4
        cv = MISOTimeSeriesSplit(n_splits=n, gap_hours=24)
        assert len(list(cv.split_by_date(hourly_df))) == n

    def test_unsorted_index_raises(self, hourly_df):
        cv = MISOTimeSeriesSplit(n_splits=3, gap_hours=24)
        shuffled = hourly_df.sample(frac=1, random_state=0)
        with pytest.raises(ValueError, match="sorted ascending"):
            list(cv.split_by_date(shuffled))

    def test_gap_24h_is_24_samples_for_hourly(self, hourly_df):
        """Verifies sklearn gap parameter translates correctly to 24 hours."""
        cv = MISOTimeSeriesSplit(n_splits=3, gap_hours=24)
        for train_df, val_df in cv.split_by_date(hourly_df):
            sample_gap = (val_df.index[0] - train_df.index[-1]) / pd.Timedelta(hours=1)
            # sklearn gap=24 means 24 samples skipped; each = 1h → gap in hours = 25
            # (last train + 24 skipped + first val = 25h apart at minimum)
            assert sample_gap >= 24
