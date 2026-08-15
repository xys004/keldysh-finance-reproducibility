"""Plot the normalization-free memory amplification from experiment 12."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "exp12_fano_memory_null.json"
PNG = ROOT / "output" / "figuras" / "fig12_fano_memory_null.png"
PDF = ROOT / "output" / "figuras" / "fig12_fano_memory_null.pdf"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = payload["results"]
    labels = [row["asset"] for row in rows]
    ratio = np.array([row["variance_ratio_to_null_median"] for row in rows])
    lower = np.array([
        row["fano_observed"] / row["null_fano_q975"] for row in rows
    ])
    upper = np.array([
        row["fano_observed"] / row["null_fano_q025"] for row in rows
    ])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    ax.errorbar(
        x, ratio, yerr=np.vstack([ratio - lower, upper - ratio]),
        fmt="o", color="#185FA5", ecolor="#6BAED6", capsize=4,
        markersize=7, linewidth=1.5,
    )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0,
               label="memory-destroying null")
    ax.axhline(2.0, color="#D95F02", linestyle=":", linewidth=1.2,
               label="twice the null median")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$F_{\rm obs}/\mathrm{median}(F_{\rm null})$")
    ax.set_ylim(0.8, max(4.6, float(upper.max()) + 0.2))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    print(f"Wrote {PNG} and {PDF}")


if __name__ == "__main__":
    main()
