"""
Time-series cross-validation with a mandatory gap between train and validation.

MISO domain requirement: predictions must be made at least 24 hours ahead
of the operating hour, so no fold may have train data within 24 h of val data.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class MISOTimeSeriesSplit:
    """
    Wraps sklearn TimeSeriesSplit to enforce the 24-hour gap rule.

    Uses sklearn's built-in ``gap`` parameter (number of samples to skip
    between train and val ends). For hourly data, gap=24 skips 24 samples.

    Parameters
    ----------
    n_splits : int
        Number of cross-validation folds.
    gap_hours : int
        Minimum hours between last train sample and first val sample.
        Assumes hourly data; each sample = 1 hour.
    test_size : int | None
        Fixed number of samples per validation fold. None = auto-sized.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap_hours: int = 24,
        test_size: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.gap_hours = gap_hours
        self.test_size = test_size
        self._tss = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap_hours,
            test_size=test_size,
        )

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, val_indices) integer arrays."""
        yield from self._tss.split(X, y, groups)

    def split_by_date(
        self, df: pd.DataFrame, y=None
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Yield (train_df, val_df) DataFrames.

        Raises AssertionError if the actual temporal gap falls below gap_hours.
        Data must be sorted ascending by index before calling.
        """
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "DataFrame index must be sorted ascending (chronological) "
                "before using MISOTimeSeriesSplit."
            )

        for train_idx, val_idx in self.split(df, y):
            train_df = df.iloc[train_idx]
            val_df   = df.iloc[val_idx]

            gap_actual = val_df.index[0] - train_df.index[-1]
            assert gap_actual >= pd.Timedelta(hours=self.gap_hours), (
                f"Gap violation: actual gap {gap_actual} < required {self.gap_hours}h. "
                "Ensure data has no missing hours (use resample/reindex to fill gaps)."
            )

            yield train_df, val_df
