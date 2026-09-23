"""Publication figure for exp15_flow_null_decomposition.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exp15_flow_null_decomposition.json"
FIGDIR = ROOT / "output" / "figuras"
MODES = ("joint", "sign", "magnitude")
LABELS = {
    "joint": "destroy sign + magnitude order",
    "sign": "destroy sign order",
    "magnitude": "destroy magnitude order",
}
COLORS = {"joint": "#374151", "sign": "#d97706", "magnitude": "#2563eb"}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    assets = [row["asset"] for row in payload["results"]]
    x = np.arange(len(assets), dtype=float)
    offsets = {"joint": -0.20, "sign": 0.0, "magnitude": 0.20}
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11,
                         "axes.labelsize": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True,
                             constrained_layout=True)
    for ax, flow_mode in zip(axes, ("raw", "normalized")):
        for null_mode in MODES:
            estimates = []
            for row in payload["results"]:
                mode = next(v for v in row["flow_definitions"]
                            if v["flow_mode"] == flow_mode)
                estimates.append(next(v for v in mode["primary_quarterly"]
                                      if v["null_mode"] == null_mode))
            y = np.array([v["variance_ratio_to_null_median"] for v in estimates])
            ci = [v["weekly_block_bootstrap"][1] for v in estimates]
            low = y - np.array([v["ratio_ci025"] for v in ci])
            high = np.array([v["ratio_ci975"] for v in ci]) - y
            ax.errorbar(
                x + offsets[null_mode], y, yerr=np.vstack([low, high]),
                fmt="o", ms=4.5, capsize=2.2, lw=1.0,
                color=COLORS[null_mode], label=LABELS[null_mode],
            )
        ax.axhline(1.0, color="0.25", lw=0.9, ls="--")
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_ylim(0.85, 16.0)
        if len(assets) > 4:
            ax.axvline(3.5, color="0.65", lw=0.8, ls=":")
            ax.text(1.5, 0.025, "original panel", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="0.35",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
            ax.text(5.5, 0.025, "post-hoc extension", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="0.35",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
        ax.set_xticks(x, assets, rotation=20, ha="right", fontsize=8.5)
        ax.set_title("(a) signed base volume" if flow_mode == "raw"
                     else "(b) volume-normalised imbalance")
        ax.set_xlabel("asset")
    axes[0].set_ylabel(r"observed / median-null weekly variance")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none",
                   framealpha=0.92, fontsize=8.5, loc="upper left")
    for suffix in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig15_flow_null_decomposition.{suffix}", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
