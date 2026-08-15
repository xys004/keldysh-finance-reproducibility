"""Higher cumulants and stratification sensitivity for the PRL supplement.

This experiment answers two distinct questions without conflating them:

1. Does the distribution of Q_T approach a Gaussian as T grows?  We report
   bias-corrected skewness gamma1=kappa3/kappa2^(3/2) and excess kurtosis
   gamma2=kappa4/kappa2^2, with circular moving-block bootstrap intervals.
   The bootstrap block spans four weeks of clock time at every T.
2. Does the weekly observed/null variance ratio depend on the calendar scale
   used to condition the permutation?  We compare quarterly, monthly and
   Monday-anchored biweekly strata, all crossed with hour of week.  Singleton
   edge strata remain fixed and their fraction is reported.

The doubled-upper-quantile screen is retained as a descriptive stress test,
not as the central decision rule.  The central claim is the normalisation-free
ratio to the null median and its empirical upper-tail probability.

Usage:
    py experiments/exp14_referee_robustness.py [null_permutations] [bootstraps]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.counting import (
    block_bootstrap_standardized_cumulants,
    net_charge,
)
from keldysh_finance.fano_validation import (
    STRATIFICATIONS,
    prepare_complete_weeks,
    stratified_fano_memory_test,
)
from keldysh_finance.flow import fetch_klines_with_flow, order_flow_imbalance


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp14_referee_robustness.json"
ASSETS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
          ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]
T_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
FOUR_WEEKS_HOURS = 4 * 168


def _cache_path(symbol: str) -> Path:
    return ROOT / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    null_permutations = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    bootstrap_replicates = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    rows = []

    for asset_index, (asset, symbol) in enumerate(ASSETS):
        df = fetch_klines_with_flow(symbol, interval="1h", years=4.0).dropna()
        eps = order_flow_imbalance(df, normalize="volume")
        cumulants = []
        for scale_index, T in enumerate(T_GRID):
            charge = net_charge(eps, T)
            block_length = max(4, int(np.ceil(FOUR_WEEKS_HOURS / T)))
            estimate = block_bootstrap_standardized_cumulants(
                charge,
                block_length=min(block_length, len(charge)),
                replicates=bootstrap_replicates,
                seed=20260815 + 10_000 * asset_index + 100 * scale_index,
                batch_size=8,
            )
            estimate["T"] = T
            cumulants.append(estimate)

        prepared = prepare_complete_weeks(df)
        nulls = []
        for scheme_index, stratification in enumerate(STRATIFICATIONS):
            estimate = stratified_fano_memory_test(
                prepared,
                permutations=null_permutations,
                seed=20260815 + 10_000 * asset_index + 1000 * scheme_index,
                batch_size=64,
                stratification=stratification,
            )
            nulls.append(estimate)
            print(
                f"{asset} {stratification:<9} "
                f"R50={estimate['variance_ratio_to_null_median']:.3f} "
                f"RQ={estimate['variance_ratio_to_null_quantile']:.3f} "
                f"p={estimate['empirical_p_upper']:.5f} "
                f"singletons={estimate['singleton_fraction']:.3%}",
                flush=True,
            )

        cache = _cache_path(symbol)
        rows.append({
            "asset": asset,
            "symbol": symbol,
            "n_hourly_candles": int(len(df)),
            "start_utc": df.index.min().isoformat(),
            "end_utc": df.index.max().isoformat(),
            "source_cache": str(cache.relative_to(ROOT)),
            "source_sha256": _sha256(cache),
            "standardized_cumulants": cumulants,
            "stratified_nulls": nulls,
        })

    payload = {
        "method": {
            "cumulants": (
                "bias-corrected standardized cumulants of non-overlapping Q_T; "
                "circular moving-block bootstrap with four-week block span"
            ),
            "null": (
                "joint within-(calendar/block period,hour-of-week) permutation; "
                "complete UTC weeks; singleton edge strata fixed and reported"
            ),
            "T_grid": T_GRID,
            "bootstrap_replicates": bootstrap_replicates,
            "null_permutations": null_permutations,
            "stratifications": list(STRATIFICATIONS),
        },
        "results": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
