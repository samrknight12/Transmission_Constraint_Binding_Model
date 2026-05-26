"""
Regime check for three flowgates showing zero val PR-AUC.

For each flowgate:
  - Monthly binding count (bar) and monthly binding rate (line) across 2023-2024
  - Annotated val-window boundary (Jul 2024)
  - Summary stats: pre-Jun vs Jul-Sep binding rate

Also prints topology observations from flowgate naming.

Output: notebooks/outputs/regime_check/{flowgate_safe_id}.png
        notebooks/outputs/regime_check/summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

FEATURES_DIR = Path("data/processed/features")
OUT_DIR      = Path("notebooks/outputs/regime_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOWGATES = {
    "TURKEY_HL-HILG FLO PRAIRIEST-WMTVERNON":  "TURKEY_HL-HILG_FLO_PRAIRIEST-WMTVERNON",
    "MORRISOT-GRANTCO FLO HANK-WAP+WAP TR2":   "MORRISOT-GRANTCO_FLO_HANK-WAP+WAP_TR2",
    "CHAR_CK-WATFORD FLO PATINGATE-CHARLIE CK": "CHAR_CK-WATFORD_FLO_PATINGATE-CHARLIE_CK",
}

VAL_START = pd.Timestamp("2024-07-01 05:00", tz="UTC")
VAL_END   = pd.Timestamp("2024-10-01 04:00", tz="UTC")

# Topology notes based on MISO naming convention:
# {FROM_BUS}-{TO_BUS} FLO {MONITORING_ELEMENT}
# Shared upstream corridor clues visible from name tokens.
TOPOLOGY_NOTES = {
    "TURKEY_HL-HILG FLO PRAIRIEST-WMTVERNON": (
        "Bus pair: TURKEY HILL – HILGER (likely SE Minnesota or N Iowa region). "
        "Monitor: PRAIRIE STREET – WMTVERNON (WMTVERNON = West Mt Vernon). "
        "No apparent token overlap with MORRISOT or CHAR_CK."
    ),
    "MORRISOT-GRANTCO FLO HANK-WAP+WAP TR2": (
        "Bus pair: MORRISON TWP – GRANT CO (W Minnesota / E South Dakota region). "
        "Monitor: HANKINSON – WAHPETON + WAHPETON TR2. "
        "HANK-WAP corridor appears in ELLIOTW-ENDERLNW FLO HANK-WAHP+WAHP T2 as well "
        "— shares the same Hankinson-Wahpeton 115 kV interface. "
        "No direct token overlap with TURKEY_HL or CHAR_CK."
    ),
    "CHAR_CK-WATFORD FLO PATINGATE-CHARLIE CK": (
        "Bus pair: CHARLIE CREEK – WATFORD CITY (W North Dakota Bakken oil patch). "
        "Monitor: PATINGATE – CHARLIE CREEK. Self-monitoring flowgate (CHAR_CK on both sides). "
        "Heavily influenced by oil-field load; binding driven by local load growth, not "
        "shared corridor with TURKEY_HL or MORRISOT."
    ),
}

CORRIDOR_SUMMARY = (
    "Topology conclusion: no shared upstream substation or corridor detected from MISO naming.\n"
    "  TURKEY_HL-HILG  : SE Minnesota/N Iowa region\n"
    "  MORRISOT-GRANTCO: W Minnesota/E South Dakota, shares HANK-WAP interface with\n"
    "                    ELLIOTW-ENDERLNW (also zero-val), suggesting a regional\n"
    "                    Hankinson-Wahpeton constraint family that binds outside Jul-Sep.\n"
    "  CHAR_CK-WATFORD : W North Dakota Bakken region, self-monitoring, load-driven.\n"
    "All three are geographically distinct — a common topology change is unlikely.\n"
    "The shared zero-val outcome is more consistent with independent seasonal patterns\n"
    "(summer inactivity) than a single upstream network event."
)


def _safe_label(name: str) -> str:
    return name[:55] + "…" if len(name) > 55 else name


def _load(safe_id: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{safe_id}.parquet"
    df = pd.read_parquet(path, columns=["binding"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly binding count and rate, filtered to 2023-2024."""
    df = df[df.index.year.isin([2023, 2024])]
    monthly = df["binding"].resample("MS").agg(["sum", "count"])
    monthly.columns = ["n_binding", "n_hours"]
    monthly["rate"] = monthly["n_binding"] / monthly["n_hours"].replace(0, np.nan)
    monthly.index = monthly.index.tz_localize(None)
    return monthly


def _plot_one(name: str, safe_id: str, monthly: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    months = monthly.index
    x = np.arange(len(months))
    labels = [m.strftime("%b %Y") for m in months]

    # Colour bars by period
    colors = []
    for m in months:
        m_utc = pd.Timestamp(m, tz="UTC")
        if VAL_START <= m_utc <= VAL_END:
            colors.append("#f28b30")   # val window = orange
        elif m_utc < VAL_START:
            colors.append("#4c8cbf")   # train = blue
        else:
            colors.append("#aaaaaa")   # post-val = grey

    bars = ax1.bar(x, monthly["n_binding"], color=colors, alpha=0.85, label="Binding hours (left)")
    ax1.set_ylabel("Binding hours / month", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(x, monthly["rate"] * 100, color="#c0392b", linewidth=2,
             marker="o", markersize=5, label="Binding rate % (right)")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax2.set_ylabel("Binding rate (%)", fontsize=11, color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    # Val-window shading
    val_months = [i for i, m in enumerate(months)
                  if VAL_START <= pd.Timestamp(m, tz="UTC") <= VAL_END]
    if val_months:
        ax1.axvspan(val_months[0] - 0.4, val_months[-1] + 0.4,
                    alpha=0.12, color="#f28b30", label="Val window (Jul-Sep 2024)")

    # Train/Val boundary
    boundary_candidates = [i for i, m in enumerate(months)
                           if pd.Timestamp(m, tz="UTC") >= VAL_START]
    if boundary_candidates:
        bx = boundary_candidates[0] - 0.5
        ax1.axvline(bx, color="black", linestyle="--", linewidth=1.2,
                    label="Train/Val boundary")

    # Pre-val vs val rate annotation
    pre_val = monthly[monthly.index < VAL_START.tz_localize(None)]
    val_win = monthly[
        (monthly.index >= VAL_START.tz_localize(None)) &
        (monthly.index <= VAL_END.tz_localize(None))
    ]
    pre_rate = pre_val["n_binding"].sum() / max(pre_val["n_hours"].sum(), 1) * 100
    val_rate  = val_win["n_binding"].sum() / max(val_win["n_hours"].sum(), 1) * 100

    subtitle = (
        f"Pre-Jul 2024 binding rate: {pre_rate:.2f}%  |  "
        f"Jul-Sep 2024 binding rate: {val_rate:.2f}%  |  "
        f"Val total: {int(val_win['n_binding'].sum())} binding hours"
    )

    ax1.set_title(f"{_safe_label(name)}\nMonthly Binding Analysis 2023-2024", fontsize=12, pad=10)
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="#444444")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUT_DIR / f"{safe_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Pre-val rate: {pre_rate:.2f}%  |  Val rate: {val_rate:.2f}%  |  Val bindings: {int(val_win['n_binding'].sum())}")


def main() -> None:
    summary_lines = ["=" * 70, "REGIME CHECK — Monthly Binding Analysis", "=" * 70, ""]

    for name, safe_id in FLOWGATES.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        df = _load(safe_id)
        monthly = _monthly_stats(df)

        print(monthly[["n_binding", "n_hours", "rate"]].to_string())
        _plot_one(name, safe_id, monthly)

        summary_lines.append(f"Flowgate: {name}")
        summary_lines.append(f"Safe ID : {safe_id}")
        summary_lines.append("")
        summary_lines.append("  Monthly binding (2023-2024):")
        for idx, row in monthly.iterrows():
            m_utc = pd.Timestamp(idx, tz="UTC")
            tag = " <-- VAL" if VAL_START <= m_utc <= VAL_END else ""
            summary_lines.append(
                f"    {idx.strftime('%b %Y')}: {int(row['n_binding']):>4} hrs  "
                f"rate={row['rate']*100:5.2f}%{tag}"
            )

        summary_lines.append("")
        summary_lines.append("  Topology note:")
        for line in TOPOLOGY_NOTES[name].split(". "):
            if line:
                summary_lines.append(f"    {line.strip()}.")
        summary_lines.append("")

    summary_lines += ["", "-" * 70, "TOPOLOGY / CORRIDOR ANALYSIS", "-" * 70, ""]
    summary_lines.append(CORRIDOR_SUMMARY)

    txt_path = OUT_DIR / "summary.txt"
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nSummary saved: {txt_path}")


if __name__ == "__main__":
    main()
