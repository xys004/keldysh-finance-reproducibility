"""Publication figure for the exact sign--size attribution."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exp17_sign_size_coupling.json"
FIGDIR = ROOT / "output" / "figuras"
COLORS = {"sign": "#2563eb", "coupling": "#d97706"}


def _bootstrap8(result: dict) -> dict:
    return next(
        row for row in result["paired_weekly_block_bootstrap"]
        if row["block_length_weeks"] == 8
    )


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["results"]
    assets = [row["asset"] for row in rows]
    x = np.arange(len(assets), dtype=float)
    width = 0.34
    plt.rcParams.update({
        "font.size": 10.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 3.25), constrained_layout=True
    )

    ax = axes[0]
    for offset, mode, label, color in (
        (-width / 2, "raw", "raw", "#374151"),
        (width / 2, "normalized", "normalised", "#7c3aed"),
    ):
        results = [row["quarterly"][mode] for row in rows]
        y = np.array([result["amplification_over_marginal"] for result in results])
        boot = [_bootstrap8(result) for result in results]
        low = y - np.array([item["amplification_ci025"] for item in boot])
        high = np.array([item["amplification_ci975"] for item in boot]) - y
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack([low, high]),
            fmt="o",
            ms=4.0,
            capsize=2.0,
            lw=1.0,
            color=color,
            label=label,
        )
    ax.axhline(1.0, color="0.35", ls="--", lw=0.8)
    ax.set_ylabel("weekly variance / marginal reference")
    ax.set_title("(a) total amplification")
    ax.legend(frameon=False, loc="upper right")

    for ax, mode, title in zip(
        axes[1:],
        ("raw", "normalized"),
        ("(b) signed base volume", "(c) normalised imbalance"),
    ):
        results = [row["quarterly"][mode] for row in rows]
        boot = [_bootstrap8(result) for result in results]
        sign = np.array([result["sign_share_of_excess"] for result in results])
        coupling = np.array([
            result["coupling_share_of_excess"] for result in results
        ])
        sign_low = sign - np.array([item["sign_share_ci025"] for item in boot])
        sign_high = np.array([item["sign_share_ci975"] for item in boot]) - sign
        coupling_low = coupling - np.array([
            item["coupling_share_ci025"] for item in boot
        ])
        coupling_high = np.array([
            item["coupling_share_ci975"] for item in boot
        ]) - coupling
        ax.bar(
            x - width / 2,
            sign,
            width,
            color=COLORS["sign"],
            label="sign order",
            yerr=np.vstack([sign_low, sign_high]),
            error_kw={"lw": 0.8, "capsize": 1.8, "ecolor": "#172554"},
        )
        ax.bar(
            x + width / 2,
            coupling,
            width,
            color=COLORS["coupling"],
            label="sign--size coupling",
            yerr=np.vstack([coupling_low, coupling_high]),
            error_kw={"lw": 0.8, "capsize": 1.8, "ecolor": "#78350f"},
        )
        ax.axhline(0.5, color="0.25", ls="--", lw=0.8)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("fraction of variance excess")
        ax.set_title(title)
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.axvline(3.5, color="0.65", lw=0.8, ls=":")
        ax.set_xticks(x, assets, rotation=28, ha="right")
        ax.text(
            1.5, -0.27, "primary", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.0, color="0.35"
        )
        ax.text(
            5.5, -0.27, "post-hoc", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.0, color="0.35"
        )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig17_sign_size_coupling.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
