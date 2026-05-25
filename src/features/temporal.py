"""
Temporal features derived purely from a UTC DatetimeIndex.

All local-time logic uses America/Chicago (MISO footprint).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# HE7–HE22 on weekdays = on-peak per MISO tariff definition
_PEAK_HOURS: frozenset[int] = frozenset(range(7, 23))
_SHOULDER_HOURS: frozenset[int] = frozenset({6, 23})

_holiday_cache: dict[tuple[int, int], pd.DatetimeIndex] = {}


def _nerc_holidays(start_year: int, end_year: int) -> pd.DatetimeIndex:
    key = (start_year, end_year)
    if key not in _holiday_cache:
        cal = USFederalHolidayCalendar()
        _holiday_cache[key] = cal.holidays(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
        )
    return _holiday_cache[key]


def build_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build temporal feature columns from a UTC DatetimeIndex.

    Cyclic sin/cos encodings are used for hour, day-of-week, and month so
    the model sees periodicity without arbitrary ordinal gaps.

    Returns a DataFrame with the same index.
    """
    local = index.tz_convert("America/Chicago")

    hour = local.hour.to_numpy(dtype=np.int8)
    dow = local.dayofweek.to_numpy(dtype=np.int8)   # 0=Mon … 6=Sun
    month = local.month.to_numpy(dtype=np.int8)

    hour_rad = 2 * np.pi * hour / 24
    dow_rad = 2 * np.pi * dow / 7
    month_rad = 2 * np.pi * (month - 1) / 12

    # Season: 1=winter(DJF), 2=spring(MAM), 3=summer(JJA), 4=fall(SON)
    season = ((month.astype(np.int16) % 12) // 3 + 1).astype(np.int8)

    # Holidays — compare naive local calendar dates
    years = local.year
    holidays = _nerc_holidays(int(years.min()), int(years.max()))
    local_dates_naive = pd.DatetimeIndex(local.date)  # tz-naive date comparison
    is_holiday = local_dates_naive.isin(holidays).astype(np.int8)

    is_weekend = (dow >= 5).astype(np.int8)
    is_peak = np.array(
        [int(h in _PEAK_HOURS and dow[i] < 5 and not is_holiday[i])
         for i, h in enumerate(hour)],
        dtype=np.int8,
    )
    is_shoulder = np.array(
        [int(h in _SHOULDER_HOURS and dow[i] < 5 and not is_holiday[i])
         for i, h in enumerate(hour)],
        dtype=np.int8,
    )

    return pd.DataFrame(
        {
            "hour_of_day":      hour,
            "hour_sin":         np.sin(hour_rad),
            "hour_cos":         np.cos(hour_rad),
            "day_of_week":      dow,
            "dow_sin":          np.sin(dow_rad),
            "dow_cos":          np.cos(dow_rad),
            "month":            month,
            "month_sin":        np.sin(month_rad),
            "month_cos":        np.cos(month_rad),
            "season":           season,
            "is_weekend":       is_weekend,
            "is_peak_hour":     is_peak,
            "is_shoulder_hour": is_shoulder,
            "is_nerc_holiday":  is_holiday,
        },
        index=index,
    )
