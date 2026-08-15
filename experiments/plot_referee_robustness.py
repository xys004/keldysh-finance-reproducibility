"""Publication figures for exp14_referee_robustness.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exp14_referee_robustness.json"
FIGDIR = ROOT / "output" / "figuras"
COLORS = {"BTC": "#d97706", "ETH": "#2563eb", "BNB": "#ca8a04", "SOL": "#059669"}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    FIGDIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    for row in payload["results"]:
        asset = row["asset"]
        data = row["standardized_cumulants"]
        T = np.array([d["T"] for d in data], float)
        for ax, key, ci_key, ylabel in [
            (axes[0], "gamma1", "gamma1_ci95", r"skewness $\gamma_1$"),
            (axes[1], "gamma2", "gamma2_ci95", r"excess kurtosis $\gamma_2$"),
        ]:
            y = np.array([d[key] for d in data], float)
            ci = np.array([d[ci_key] for d in data], float)
            ax.plot(T, y, marker="o", ms=3.5, lw=1.3, color=COLORS[asset], label=asset)
            ax.fill_between(T, ci[:, 0], ci[:, 1], color=COLORS[asset], alpha=0.12)
            ax.set_xscale("log", base=2)
            ax.set_xlabel(r"counting window $T$ (hours)")
            ax.set_ylabel(ylabel)
            ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    axes[0].set_title("(a) asymmetry does not vanish uniformly")
    axes[1].set_title("(b) tails remain non-Gaussian")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig13_standardized_cumulants.{suffix}", dpi=220)
    plt.close(fig)

    schemes = payload["method"]["stratifications"]
    x = np.arange(len(schemes), dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.25), constrained_layout=True)
    offsets = np.linspace(-0.27, 0.27, len(payload["results"]))
    for offset, row in zip(offsets, payload["results"]):
        values = [
            next(d for d in row["stratified_nulls"] if d["stratification"] == scheme)
            ["variance_ratio_to_null_median"]
            for scheme in schemes
        ]
        ax.plot(x + offset, values, marker="o", lw=1.2, color=COLORS[row["asset"]],
                label=row["asset"])
    ax.axhline(2.0, color="0.2", lw=1.0, ls="--", label="twice null median")
    ax.set_xticks(x, ["quarterly", "monthly", "biweekly"])
    ax.set_ylabel(r"$F_{\rm obs}/\mathrm{median}(F_{\rm null})$")
    ax.set_title("Conditioning reveals the scale dependence of the excess")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig14_stratification_sensitivity.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
