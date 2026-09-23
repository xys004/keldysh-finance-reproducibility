"""Exact, stratum-aware attribution of weekly signed-flow variance.

The experiment replaces the asymmetric interpretation of three permutation
nulls with an additive finite-sample decomposition.  It retains the observed
diagonal second moment, measures sign ordering after centring signs and
homogenising magnitudes within each calendar stratum, and assigns the exact
remainder to sign--size coupling.  Paired circular-block bootstrap intervals
quantify uncertainty in both the shares and the raw-versus-normalised change.

Usage:
    python experiments/exp17_sign_size_coupling.py [bootstrap_replicates]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.fano_validation import (
    paired_normalization_share_difference,
    prepare_complete_weeks,
    stratified_sign_size_decomposition,
)
from keldysh_finance.flow import fetch_klines_with_flow


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp17_sign_size_coupling.json"
CORE_ASSETS = ("BTC", "ETH", "BNB", "SOL")
ASSETS = (
    ("BTC", "BTCUSDT"),
    ("ETH", "ETHUSDT"),
    ("BNB", "BNBUSDT"),
    ("SOL", "SOLUSDT"),
    ("XRP", "XRPUSDT"),
    ("ADA", "ADAUSDT"),
    ("DOGE", "DOGEUSDT"),
    ("AVAX", "AVAXUSDT"),
)


def _cache_path(symbol: str) -> Path:
    return ROOT / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eight_week(result: dict) -> dict:
    return next(
        row for row in result["paired_weekly_block_bootstrap"]
        if row["block_length_weeks"] == 8
    )


def main() -> None:
    replicates = int(sys.argv[1]) if len(sys.argv) > 1 else 3999
    rows = []
    for asset_index, (asset, symbol) in enumerate(ASSETS):
        frame = fetch_klines_with_flow(symbol, interval="1h", years=4.0).dropna()
        prepared = {
            mode: prepare_complete_weeks(frame, flow_mode=mode)
            for mode in ("raw", "normalized")
        }
        quarterly = {}
        sensitivity = {}
        for mode_index, mode in enumerate(("raw", "normalized")):
            seed = 20260816 + 100_000 * asset_index + 10_000 * mode_index
            quarterly[mode] = stratified_sign_size_decomposition(
                prepared[mode],
                stratification="quarterly",
                bootstrap_replicates=replicates,
                seed=seed,
            )
            sensitivity[mode] = []
            if asset in CORE_ASSETS:
                for scheme_index, scheme in enumerate(("monthly", "biweekly")):
                    sensitivity[mode].append(
                        stratified_sign_size_decomposition(
                            prepared[mode],
                            stratification=scheme,
                            bootstrap_replicates=replicates,
                            seed=seed + 1_000 * (scheme_index + 1),
                        )
                    )

        paired = paired_normalization_share_difference(
            prepared["raw"],
            prepared["normalized"],
            stratification="quarterly",
            bootstrap_replicates=replicates,
            seed=20260817 + 100_000 * asset_index,
        )
        paired_sensitivity = []
        if asset in CORE_ASSETS:
            for scheme_index, scheme in enumerate(("monthly", "biweekly")):
                paired_sensitivity.append(
                    paired_normalization_share_difference(
                        prepared["raw"],
                        prepared["normalized"],
                        stratification=scheme,
                        bootstrap_replicates=replicates,
                        seed=(20260818 + 100_000 * asset_index
                              + 1_000 * scheme_index),
                    )
                )

        rows.append({
            "asset": asset,
            "symbol": symbol,
            "panel": "primary" if asset in CORE_ASSETS else "posthoc",
            "start_utc": frame.index.min().isoformat(),
            "end_utc": frame.index.max().isoformat(),
            "source_cache": str(_cache_path(symbol).relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "source_sha256": _sha256(_cache_path(symbol)),
            "quarterly": quarterly,
            "stratification_sensitivity": sensitivity,
            "paired_normalization_effect": paired,
            "paired_normalization_sensitivity": paired_sensitivity,
        })
        for mode in ("raw", "normalized"):
            result = quarterly[mode]
            ci = _eight_week(result)
            print(
                f"{asset:<4} {mode:<10} amplification="
                f"{result['amplification_over_marginal']:.3f} "
                f"sign={result['sign_share_of_excess']:.3f} "
                f"coupling={result['coupling_share_of_excess']:.3f} "
                f"CI8=[{ci['coupling_share_ci025']:.3f},"
                f"{ci['coupling_share_ci975']:.3f}]",
                flush=True,
            )

    core = [row for row in rows if row["panel"] == "primary"]
    claims = {
        "point_coupling_majority_all_primary_raw": all(
            row["quarterly"]["raw"]["coupling_share_of_excess"] > 0.5
            for row in core
        ),
        "ci8_coupling_majority_all_primary_raw": all(
            _eight_week(row["quarterly"]["raw"])["coupling_share_ci025"] > 0.5
            for row in core
        ),
        "point_sign_share_increases_after_normalization_all_assets": all(
            row["paired_normalization_effect"]["delta_sign_share"] > 0.0
            for row in rows
        ),
        "ci8_sign_share_increase_all_primary": all(
            _eight_week(row["paired_normalization_effect"])[
                "delta_sign_share_ci025"
            ] > 0.0
            for row in core
        ),
    }
    payload = {
        "method": {
            "flow_residual": (
                "signed flow demeaned within period x UTC hour-of-week strata"
            ),
            "marginal_reference": (
                "sum_t x_t^2/(W-1), the exact expectation of the joint-order "
                "permutation variance for complete weeks"
            ),
            "sign_order_component": (
                "off-diagonal term after centring signs and replacing magnitudes "
                "by their means within the same stratum"
            ),
            "sign_size_coupling_component": (
                "exact remainder required to reconstruct observed weekly variance"
            ),
            "interpretive_boundary": (
                "declared fixed-diagonal, non-orthogonal attribution; its "
                "reference plus sign term is not the ordinary variance of "
                "the homogenised proxy, and it is not a unique causal "
                "allocation of an interaction"
            ),
            "bootstrap": (
                "paired circular moving-block resampling of weekly component "
                "vectors; stratum estimates held fixed"
            ),
            "bootstrap_replicates": replicates,
            "bootstrap_block_lengths_weeks": [4, 8, 13, 26],
            "primary_assets": list(CORE_ASSETS),
            "posthoc_assets": [a for a, _ in ASSETS if a not in CORE_ASSETS],
        },
        "results": rows,
        "claim_checks": claims,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(claims, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
