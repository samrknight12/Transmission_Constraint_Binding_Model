"""
Download 2 years (2023-2024) of MISO market data.

Report URLs (verified 2026-05-24):
  da_bc       — https://docs.misoenergy.org/marketreports/2023_da_bc_HIST.csv
                 Annual HIST file; 2 downloads cover the full 2-year window.
  load        — https://docs.misoenergy.org/marketreports/DA_Load_EPNodes_YYYYMMDD.zip
  lmp         — https://docs.misoenergy.org/marketreports/YYYYMMDD_da_exante_lmp.csv

Run:
    py scripts/download_miso_data.py
"""
from __future__ import annotations

import os
import time
import logging
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL  = "https://docs.misoenergy.org/marketreports"
SESSION   = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (research/data-pipeline)"
MAX_WORKERS   = 6   # concurrent downloads — respectful of MISO servers
MAX_RETRIES   = 3
RETRY_DELAY_S = 5

START_DATE = date(2023, 1, 1)
END_DATE   = date(2024, 12, 31)


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_one(url: str, dest: Path, retries: int = MAX_RETRIES) -> str:
    """
    Download `url` to `dest`.  Skips if file already exists.
    Returns a status string: 'skipped', 'ok', or 'missing' / error message.
    """
    if dest.exists():
        return "skipped"

    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException as exc:
            if attempt == retries:
                return f"error:{exc}"
            time.sleep(RETRY_DELAY_S * attempt)
            continue

        if r.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return "ok"
        elif r.status_code == 404:
            return "missing"
        else:
            if attempt == retries:
                return f"http:{r.status_code}"
            time.sleep(RETRY_DELAY_S)

    return "error:exhausted"


def _run_parallel(tasks: list[tuple[str, Path]], desc: str) -> dict[str, int]:
    """Run a list of (url, dest) download tasks concurrently. Returns status counts."""
    counts: dict[str, int] = {}
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, url, dest): (url, dest) for url, dest in tasks}
        done = 0
        for fut in as_completed(futures):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 50 == 0 or done == total:
                log.info("%s: %d/%d  %s", desc, done, total, counts)

    return counts


# ── Report-specific download functions ───────────────────────────────────────

def download_da_bc_hist(out_dir: Path = Path("data/raw/binding_constraints")) -> None:
    """
    Download annual DA binding constraint HIST CSVs.
    2 files cover all of 2023-2024.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(START_DATE.year, END_DATE.year + 1):
        filename = f"{year}_da_bc_HIST.csv"
        url  = f"{BASE_URL}/{filename}"
        dest = out_dir / filename
        status = _download_one(url, dest)
        size = dest.stat().st_size if dest.exists() else 0
        log.info("da_bc HIST %d — %s  (%.1f MB)", year, status, size / 1e6)


def download_load_forecasts(out_dir: Path = Path("data/raw/load_forecasts")) -> None:
    """Download daily DA_Load_EPNodes ZIP files for every date in range."""
    tasks = []
    d = START_DATE
    while d <= END_DATE:
        ds   = d.strftime("%Y%m%d")
        url  = f"{BASE_URL}/DA_Load_EPNodes_{ds}.zip"
        dest = out_dir / f"DA_Load_EPNodes_{ds}.zip"
        tasks.append((url, dest))
        d += timedelta(days=1)

    counts = _run_parallel(tasks, "load_forecast")
    log.info("load_forecast final: %s", counts)


def download_da_exante_lmp(out_dir: Path = Path("data/raw/lmp")) -> None:
    """Download daily DA ex-ante LMP CSV files for every date in range."""
    tasks = []
    d = START_DATE
    while d <= END_DATE:
        ds   = d.strftime("%Y%m%d")
        url  = f"{BASE_URL}/{ds}_da_exante_lmp.csv"
        dest = out_dir / f"{ds}_da_exante_lmp.csv"
        tasks.append((url, dest))
        d += timedelta(days=1)

    counts = _run_parallel(tasks, "da_exante_lmp")
    log.info("da_exante_lmp final: %s", counts)


def download_gen_fuel_mix(out_dir: Path = Path("data/raw/gen_fuel_mix")) -> None:
    """
    Download annual Historical Generation Fuel Mix XLSX files.

    Columns: Market Date, HourEnding, Region, Fuel Type,
             DA Cleared UDS Generation, RT Generation State Estimator.

    Fuel types include Wind and Solar — DA Cleared MW is the best available
    public proxy for the DA wind/solar forecast (what the market cleared).
    Thermal rows (Coal, Gas, Nuclear) serve as an outage-proxy feature:
    sustained drops below rolling average indicate forced/planned outages.

    NOTE: MISO's CROW system (referenced in CLAUDE.md) provides true planned
    and forced outage data but is not publicly available. These fuel mix
    files are the closest public substitute.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(START_DATE.year, END_DATE.year + 1):
        filename = f"historical_gen_fuel_mix_{year}.xlsx"
        url  = f"{BASE_URL}/{filename}"
        dest = out_dir / filename
        status = _download_one(url, dest)
        size = dest.stat().st_size if dest.exists() else 0
        log.info("gen_fuel_mix %d — %s  (%.1f MB)", year, status, size / 1e6)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MISO market data 2023-2024")
    parser.add_argument(
        "--reports",
        nargs="+",
        choices=["da_bc", "load", "lmp", "gen_fuel_mix", "all"],
        default=["all"],
        help="Which report types to download (default: all)",
    )
    args = parser.parse_args()
    reports = set(args.reports)
    if "all" in reports:
        reports = {"da_bc", "load", "lmp", "gen_fuel_mix"}

    if "da_bc" in reports:
        log.info("=== Downloading da_bc HIST (2 files) ===")
        download_da_bc_hist()

    if "load" in reports:
        log.info("=== Downloading DA load forecasts (%d files) ===",
                 (END_DATE - START_DATE).days + 1)
        download_load_forecasts()

    if "lmp" in reports:
        log.info("=== Downloading DA ex-ante LMP (%d files) ===",
                 (END_DATE - START_DATE).days + 1)
        download_da_exante_lmp()

    if "gen_fuel_mix" in reports:
        log.info("=== Downloading Historical Gen Fuel Mix (2 files — wind/solar + outage proxy) ===")
        download_gen_fuel_mix()

    log.info("Done.")
