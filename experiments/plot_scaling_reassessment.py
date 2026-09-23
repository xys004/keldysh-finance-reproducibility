"""Publication figure for the asset-specific scaling reassessment."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exp16_scaling_reassessment.json"
FIGDIR = ROOT / "output" / "figuras"
COLORS = {"BTC": "#d97706", "ETH": "#2563eb", "BNB": "#ca8a04", "SOL": "#059669"}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

    ax = axes[0]
    rows = payload["single_power_series_tests"]
    for row in rows:
        asset, interval = row["series"].split("|")
        marker = "o" if interval == "1h" else "s"
        ax.errorbar(
            row["single_acf_prediction"], row["observed_block_variance_exponent"],
            yerr=row["observed_standard_error"], fmt=marker, ms=4.5,
            capsize=2, color=COLORS[asset],
        )
    limits = (1.15, 1.95)
    ax.plot(limits, limits, color="0.25", ls="--", lw=0.9)
    ax.set(xlim=limits, ylim=limits,
           xlabel=r"single-power prediction $2-a$",
           ylabel=r"measured block exponent $\nu$")
    ax.set_title("(a) one power fails in all eight series")

    ax = axes[1]
    assets = [row["asset"] for row in payload["two_regime_asset_checks"]]
    y = np.arange(len(assets))[::-1]
    for pos, row in zip(y, payload["two_regime_asset_checks"]):
        low, high = row["finite_scale_consistency_band"]
        ax.plot([low, high], [pos, pos], color=COLORS[row["asset"]], lw=6,
                alpha=0.28, solid_capstyle="butt")
        points = {v["series"].split("|")[1]: v for v in row["block_exponents"]}
        ax.plot(points["1h"]["nu"], pos + 0.09, "o", color=COLORS[row["asset"]],
                ms=4.5, label="1 h" if pos == y[0] else None)
        ax.plot(points["4h"]["nu"], pos - 0.09, "s", color=COLORS[row["asset"]],
                ms=4.2, label="4 h" if pos == y[0] else None)
    ax.axvline(1.0, color="0.25", ls="--", lw=0.9)
    ax.set_yticks(y, assets)
    ax.set_xlabel(r"variance exponent $\nu$")
    ax.set_title("(b) two-regime consistency bands")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    for suffix in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig16_scaling_reassessment.{suffix}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
