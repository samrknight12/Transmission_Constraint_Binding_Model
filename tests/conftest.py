"""
Shared pytest fixtures for MISO model tests.

All synthetic data uses a 48-hour UTC window to keep tests fast.
The ~20:1 class imbalance is intentionally reflected in shadow_price_df.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FLOWGATE_ID = "TEST.GATE 345 KV"


@pytest.fixture
def utc_index():
    """48-hour UTC DatetimeIndex at hourly frequency."""
    return pd.date_range("2024-06-15 00:00", periods=48, freq="h", tz="UTC")


@pytest.fixture
def shadow_price_df(utc_index):
    rng = np.random.default_rng(42)
    # Exponential distribution: most values < 0.01 → realistic imbalance
    prices = rng.exponential(scale=0.008, size=len(utc_index))
    return pd.DataFrame(
        {
            "datetime":    utc_index,
            "flowgate_id": FLOWGATE_ID,
            "shadow_price": prices,
        }
    )


@pytest.fixture
def load_forecast_df(utc_index):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"load_forecast_mw": 22_000 + rng.normal(0, 600, size=len(utc_index))},
        index=utc_index,
    )


@pytest.fixture
def renewable_df(utc_index):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "wind_forecast_mw":  rng.uniform(500, 3_500, size=len(utc_index)),
            "solar_forecast_mw": rng.uniform(0,   1_500, size=len(utc_index)),
        },
        index=utc_index,
    )


@pytest.fixture
def outage_df(utc_index):
    start = utc_index[4]
    end   = utc_index[16]
    return pd.DataFrame(
        {
            "resource_id":  ["UNIT_A",   "UNIT_B"],
            "outage_start": [start,      start],
            "outage_end":   [end,        end],
            "capacity_mw":  [500.0,      300.0],
            "outage_type":  ["planned",  "forced"],
        }
    )


@pytest.fixture
def ptdf_df():
    return pd.DataFrame(
        {
            "flowgate_id": [FLOWGATE_ID, FLOWGATE_ID],
            "resource_id": ["UNIT_A",    "UNIT_B"],
            "ptdf_value":  [0.15,        -0.08],
        }
    )


@pytest.fixture
def flowgate_loading_df(utc_index):
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "flowgate_id": FLOWGATE_ID,
            "loading_pct": rng.uniform(30, 95, size=len(utc_index)),
        },
        index=utc_index,
    )
