"""Parallel flow definitions and non-additive permutation controls.

The primary analysis uses complete UTC weeks and quarterly x hour-of-week
strata.  For raw signed volume and volume-normalised imbalance separately it
compares three nulls: joint signed-flow permutation, sign permutation with the
magnitude path fixed, and magnitude permutation with the sign path fixed.  The
latter two are asymmetric controls and are not interpreted as additive variance
components; experiment 17 provides the exact attribution.
Monthly and biweekly joint nulls provide stratification sensitivity.  Moving-
block bootstrap intervals quantify uncertainty in the observed/null-median
weekly variance ratio, conditional on each empirical null distribution.

Usage:
    python experiments/exp15_flow_null_decomposition.py [permutations] [bootstraps]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.fano_validation import (
    FLOW_MODES,
    NULL_MODES,
    prepare_complete_weeks,
    stratified_variance_memory_test,
)
from keldysh_finance.flow import fetch_klines_with_flow


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp15_flow_null_decomposition.json"
CORE_ASSETS = ("BTC", "ETH", "BNB", "SOL")
ASSETS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
          ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT"),
          ("XRP", "XRPUSDT"), ("ADA", "ADAUSDT"),
          ("DOGE", "DOGEUSDT"), ("AVAX", "AVAXUSDT")]


def _cache_path(symbol: str) -> Path:
    return ROOT / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation_summary(series: dict[str, pd.Series]) -> dict:
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    weekly = aligned.resample("W-MON", label="left", closed="left").sum()
    return {
        "assets": list(aligned.columns),
        "n_common_hours": int(len(aligned)),
        "hourly_pearson": aligned.corr().to_numpy(float).tolist(),
        "n_calendar_weeks": int(len(weekly)),
        "weekly_sum_pearson": weekly.corr().to_numpy(float).tolist(),
    }


def main() -> None:
    permutations = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    bootstraps = int(sys.argv[2]) if len(sys.argv) > 2 else 3999
    resume = "--resume" in sys.argv[3:]
    sensitivity_permutations = min(permutations, 1999)
    rows = []
    existing: dict[str, dict] = {}
    if resume and OUT.exists():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        method = previous.get("method", {})
        if (method.get("permutations") == permutations
                and method.get("bootstrap_replicates") == bootstraps):
            existing = {row["asset"]: row for row in previous.get("results", [])}
        else:
            raise ValueError("cannot resume from output with different replicate counts")
    correlation_series: dict[str, dict[str, pd.Series]] = {
        mode: {} for mode in FLOW_MODES
    }

    for asset_index, (asset, symbol) in enumerate(ASSETS):
        frame = fetch_klines_with_flow(symbol, interval="1h", years=4.0).dropna()
        if asset in existing:
            for flow_mode in FLOW_MODES:
                prepared = prepare_complete_weeks(frame, flow_mode=flow_mode)
                correlation_series[flow_mode][asset] = pd.Series(
                    prepared.flow, index=prepared.index
                )
            rows.append(existing[asset])
            print(f"{asset}: reused completed analysis", flush=True)
            continue
        mode_results = []
        for mode_index, flow_mode in enumerate(FLOW_MODES):
            prepared = prepare_complete_weeks(frame, flow_mode=flow_mode)
            correlation_series[flow_mode][asset] = pd.Series(
                prepared.flow, index=prepared.index
            )
            primary = []
            for null_index, null_mode in enumerate(NULL_MODES):
                seed = (20260815 + 100_000 * asset_index
                        + 10_000 * mode_index + 1_000 * null_index)
                estimate = stratified_variance_memory_test(
                    prepared,
                    permutations=permutations,
                    seed=seed,
                    batch_size=256,
                    stratification="quarterly",
                    null_mode=null_mode,
                    bootstrap_replicates=bootstraps,
                )
                primary.append(estimate)
                ci = estimate["weekly_block_bootstrap"][1]
                print(
                    f"{asset} {flow_mode:<10} {null_mode:<9} "
                    f"R={estimate['variance_ratio_to_null_median']:.3f} "
                    f"CI8=[{ci['ratio_ci025']:.3f},{ci['ratio_ci975']:.3f}] "
                    f"p={estimate['empirical_p_upper']:.5f}",
                    flush=True,
                )

            sensitivity = []
            for scheme_index, scheme in enumerate(("monthly", "biweekly")):
                sensitivity.append(stratified_variance_memory_test(
                    prepared,
                    permutations=sensitivity_permutations,
                    seed=(20260815 + 100_000 * asset_index
                          + 10_000 * mode_index + 8_000 + 1_000 * scheme_index),
                    batch_size=256,
                    stratification=scheme,
                    null_mode="joint",
                ))
            mode_results.append({
                "flow_mode": flow_mode,
                "primary_quarterly": primary,
                "joint_stratification_sensitivity": sensitivity,
            })

        cache = _cache_path(symbol)
        rows.append({
            "asset": asset,
            "symbol": symbol,
            "start_utc": frame.index.min().isoformat(),
            "end_utc": frame.index.max().isoformat(),
            "source_cache": str(cache.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": _sha256(cache),
            "flow_definitions": mode_results,
        })

    gate = {}
    for flow_mode in FLOW_MODES:
        joint = []
        for row in rows:
            if row["asset"] not in CORE_ASSETS:
                continue
            mode = next(x for x in row["flow_definitions"]
                        if x["flow_mode"] == flow_mode)
            estimate = next(x for x in mode["primary_quarterly"]
                            if x["null_mode"] == "joint")
            joint.append({
                "asset": row["asset"],
                "passed": estimate["memory_majority_gate"],
                "ratio_to_null_median": estimate["variance_ratio_to_null_median"],
            })
        gate[flow_mode] = {
            "asset_results": joint,
            "all_assets_passed": bool(all(x["passed"] for x in joint)),
        }

    payload = {
        "method": {
            "asset_panel": {
                "original": list(CORE_ASSETS),
                "posthoc_extension": [asset for asset, _ in ASSETS
                                       if asset not in CORE_ASSETS],
                "extension_selection": (
                    "additional high-activity USDT pairs with a complete "
                    "four-year hourly history; selected before inspecting "
                    "their null-decomposition outcomes"
                ),
            },
            "flow_modes": {
                "raw": "2*tbBase-Volume in base-asset units",
                "normalized": "(2*tbBase-Volume)/Volume, zero when Volume=0",
            },
            "primary_stratification": "quarter x UTC hour-of-week",
            "null_modes": {
                "joint": "permute complete demeaned signed-flow values",
                "sign": "keep magnitude path; permute signs within strata",
                "magnitude": "keep sign path; permute magnitudes within strata",
            },
            "null_limitation": (
                "sign and magnitude nulls do not preserve contemporaneous "
                "sign-magnitude pairing; they are attribution diagnostics, not "
                "structural order-book models"
            ),
            "permutations": permutations,
            "stratification_sensitivity_permutations": sensitivity_permutations,
            "bootstrap_replicates": bootstraps,
            "bootstrap_block_lengths_weeks": [4, 8, 13, 26],
            "bootstrap_interpretation": (
                "uncertainty of observed weekly variance conditional on the "
                "empirical null median"
            ),
        },
        "results": rows,
        "cross_asset_dependence": {
            mode: _correlation_summary(correlation_series[mode])
            for mode in FLOW_MODES
        },
        "predeclared_all_asset_gate": gate,
        "posthoc_asset_extension": {
            "assets": [asset for asset, _ in ASSETS if asset not in CORE_ASSETS],
            "status": "robustness panel; not part of the predeclared all-asset gate",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
