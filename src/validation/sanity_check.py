"""
Sanity checks for the MISO master dataset.

Usage:
    python src/validation/sanity_check.py

Prints PASS/FAIL for each check and exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MASTER_PATH = Path("data/processed/master_dataset.parquet")

# All market timestamps are fixed EST = UTC-5 (MISO does not observe DST).
# Split boundaries expressed as UTC so they can be compared directly to the index.
#
#   Train : 2023-01-01 00:00 EST  to  2024-06-30 23:00 EST
#   Val   : 2024-07-01 00:00 EST  to  2024-09-30 23:00 EST
#   Test  : 2024-10-01 00:00 EST  to  2024-12-31 23:00 EST
#
# EST = UTC-5 (fixed offset, no DST), so midnight EST = 05:00 UTC.
_UTC = "UTC"
_DATA_START  = pd.Timestamp("2023-01-01 05:00", tz=_UTC)   # 2023-01-01 00:00 EST
_TRAIN_END   = pd.Timestamp("2024-07-01 04:00", tz=_UTC)   # 2024-06-30 23:00 EST
_VAL_START   = pd.Timestamp("2024-07-01 05:00", tz=_UTC)   # 2024-07-01 00:00 EST
_VAL_END     = pd.Timestamp("2024-10-01 04:00", tz=_UTC)   # 2024-09-30 23:00 EST
_TEST_START  = pd.Timestamp("2024-10-01 05:00", tz=_UTC)   # 2024-10-01 00:00 EST
_DATA_END    = pd.Timestamp("2025-01-01 04:00", tz=_UTC)   # 2024-12-31 23:00 EST

_results: list[bool] = []


# ── Result recorder ───────────────────────────────────────────────────────────

def _record(label: str, passed: bool, detail: str = "") -> bool:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {label}")
    if detail:
        print(f"         {detail}")
    _results.append(passed)
    return passed


# ── Individual checks ─────────────────────────────────────────────────────────

def check_no_nans(df: pd.DataFrame) -> None:
    print("\nCheck 1: No NaN values in any feature column")
    feature_cols = [c for c in df.columns if c != "flowgate_id"]
    null_counts = df[feature_cols].isna().sum()
    bad = null_counts[null_counts > 0]
    _record(
        "No NaN values across all feature columns",
        bad.empty,
        ("" if bad.empty
         else "Columns with NaN: " + ", ".join(f"{c}={n}" for c, n in bad.items())),
    )


def check_class_imbalance(df: pd.DataFrame) -> None:
    print("\nCheck 2: Target class imbalance 10:1 to 35:1 per flowgate")
    failures: list[str] = []
    for fg_id, grp in df.groupby("flowgate_id", sort=True):
        n_pos = int(grp["binding"].sum())
        n_neg = len(grp) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
        in_range = 10.0 <= ratio <= 35.0
        flag = "  <-- OUTSIDE RANGE" if not in_range else ""
        print(f"         {fg_id:<55s}  {ratio:6.1f}:1{flag}")
        if not in_range:
            failures.append(fg_id)
    _record(
        "All flowgates have imbalance in [10:1, 35:1]",
        not failures,
        (f"{len(failures)} outside range: {', '.join(failures[:5])}"
         + (" ..." if len(failures) > 5 else "")) if failures else "",
    )


def check_date_range_and_gaps(df: pd.DataFrame) -> None:
    print("\nCheck 3: Date range 2023-01-01 to 2024-12-31, no gaps > 48h")

    # Use a single flowgate's index as the representative hourly time series
    first_fg = df["flowgate_id"].iloc[0]
    idx = df[df["flowgate_id"] == first_fg].index.sort_values()

    _record(
        "Data starts on or before 2023-01-01 00:00 EST",
        idx.min() <= _DATA_START,
        f"Actual start: {idx.min()}",
    )
    _record(
        "Data ends on or after 2024-12-31 23:00 EST",
        idx.max() >= _DATA_END,
        f"Actual end: {idx.max()}",
    )

    diffs = idx.to_series().diff().dropna()
    max_gap = diffs.max()
    big_gaps = diffs[diffs > pd.Timedelta(hours=48)]
    _record(
        "No gaps larger than 48h in the hourly index",
        big_gaps.empty,
        f"Max gap: {max_gap}"
        + (f"  |  {len(big_gaps)} gap(s) > 48h found" if not big_gaps.empty else ""),
    )


def check_split_chronology(df: pd.DataFrame) -> None:
    print("\nCheck 4: Train/val/test split is strictly chronological")

    first_fg = df["flowgate_id"].iloc[0]
    idx = df[df["flowgate_id"] == first_fg].index.sort_values()

    train_idx = idx[(idx >= _DATA_START) & (idx <= _TRAIN_END)]
    val_idx   = idx[(idx >= _VAL_START)  & (idx <= _VAL_END)]
    test_idx  = idx[(idx >= _TEST_START) & (idx <= _DATA_END)]

    _record(
        "Train  : 2023-01-01 to 2024-06-30",
        len(train_idx) > 0,
        f"{len(train_idx):,} hours  "
        f"({train_idx.min()} -> {train_idx.max()})" if len(train_idx) else "0 hours",
    )
    _record(
        "Val    : 2024-07-01 to 2024-09-30",
        len(val_idx) > 0,
        f"{len(val_idx):,} hours  "
        f"({val_idx.min()} -> {val_idx.max()})" if len(val_idx) else "0 hours",
    )
    _record(
        "Test   : 2024-10-01 to 2024-12-31",
        len(test_idx) > 0,
        f"{len(test_idx):,} hours  "
        f"({test_idx.min()} -> {test_idx.max()})" if len(test_idx) else "0 hours",
    )

    # Strict ordering: last train < first val, last val < first test
    if len(train_idx) and len(val_idx):
        no_tv_overlap = train_idx.max() < val_idx.min()
        _record(
            "Train end strictly before val start",
            no_tv_overlap,
            f"Train end: {train_idx.max()}  Val start: {val_idx.min()}",
        )
    if len(val_idx) and len(test_idx):
        no_vt_overlap = val_idx.max() < test_idx.min()
        _record(
            "Val end strictly before test start",
            no_vt_overlap,
            f"Val end: {val_idx.max()}  Test start: {test_idx.min()}",
        )

    # All hours are accounted for across the three splits
    covered = len(train_idx) + len(val_idx) + len(test_idx)
    _record(
        "Splits together cover the full index without overlap",
        covered == len(idx),
        f"Split total: {covered:,}  Index total: {len(idx):,}",
    )


def check_freq_no_leakage(df: pd.DataFrame) -> None:
    """
    Verify binding_freq_trailing_30d does not include the same-hour target.

    If shift(1) was applied correctly, then at the first-ever binding hour for
    each flowgate the 30d frequency must be exactly 0.0 — no prior binding
    has been seen, so the lagged rolling mean has nothing to accumulate.
    A non-zero value at that point can only arise if the current hour was
    included in the rolling window (same-hour leakage).
    """
    print("\nCheck 5: binding_freq_trailing_30d excludes same-hour target (shift(1) check)")
    failures: list[str] = []
    for fg_id, grp in df.groupby("flowgate_id", sort=True):
        grp = grp.sort_index()
        first_binding = grp[grp["binding"] == 1]
        if first_binding.empty:
            continue
        t0 = first_binding.index[0]
        freq_at_t0 = float(grp.loc[t0, "flowgate_binding_freq_30d"])
        if freq_at_t0 != 0.0:
            print(
                f"         FAIL  {fg_id}: "
                f"freq_30d={freq_at_t0:.6f} at first binding event {t0}"
            )
            failures.append(fg_id)
    _record(
        "freq_30d == 0 at first-ever binding event for all flowgates",
        not failures,
        f"{len(failures)} flowgate(s) failed" if failures else "",
    )


def check_loading_observed_rate(df: pd.DataFrame) -> None:
    print("\nCheck 6: loading_pct_is_observed rate per flowgate (expect 3-10%)")
    out_of_range: list[str] = []
    for fg_id, grp in df.groupby("flowgate_id", sort=True):
        rate = float(grp["flowgate_loading_pct_is_observed"].mean())
        in_range = 0.03 <= rate <= 0.10
        flag = "  <-- outside 3-10%" if not in_range else ""
        print(f"         {fg_id:<55s}  {rate:.2%}{flag}")
        if not in_range:
            out_of_range.append(fg_id)
    _record(
        "All flowgates: observed loading rate within 3-10%",
        not out_of_range,
        (f"{len(out_of_range)} outside range: "
         + ", ".join(out_of_range[:5])
         + (" ..." if len(out_of_range) > 5 else "")) if out_of_range else "",
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== MISO Master Dataset Sanity Checks ===")

    if not MASTER_PATH.exists():
        print(f"\nERROR: {MASTER_PATH} not found. Run build_master_dataset.py first.")
        sys.exit(1)

    print(f"\nLoading {MASTER_PATH} ...")
    df = pd.read_parquet(MASTER_PATH)
    print(f"Shape: {df.shape}  |  Flowgates: {df['flowgate_id'].nunique()}")

    check_no_nans(df)
    check_class_imbalance(df)
    check_date_range_and_gaps(df)
    check_split_chronology(df)
    check_freq_no_leakage(df)
    check_loading_observed_rate(df)

    n_pass = sum(_results)
    n_fail = len(_results) - n_pass
    print(f"\n{'=' * 45}")
    print(f"Total: {n_pass} PASS, {n_fail} FAIL out of {len(_results)} checks")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
