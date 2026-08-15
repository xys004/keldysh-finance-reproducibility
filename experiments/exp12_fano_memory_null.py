"""Stratified, size-preserving null for the weekly counting variance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.fano_validation import (prepare_complete_weeks,
                                             stratified_fano_memory_test)
from keldysh_finance.flow import fetch_klines_with_flow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp12_fano_memory_null.json"
ASSETS = [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
          ("BNB", "BNBUSDT"), ("SOL", "SOLUSDT")]


def main() -> None:
    permutations = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    rows = []
    for i, (asset, symbol) in enumerate(ASSETS):
        df = fetch_klines_with_flow(symbol, interval="1h", years=4.0)
        prepared = prepare_complete_weeks(df)
        result = stratified_fano_memory_test(
            prepared,
            permutations=permutations,
            seed=20260814 + 1000 * i,
        )
        result.update({"asset": asset, "symbol": symbol})
        rows.append(result)
        print(
            f"{asset}: Fobs={result['fano_observed']:.1f}, "
            f"Fnull50={result['fano_null_median']:.1f}, "
            f"ratio={result['variance_ratio_to_null_median']:.3f}, "
            f"gate={result['memory_majority_gate']}",
            flush=True,
        )

    payload = {
        "method": "joint within-(quarter,hour-of-week) permutation; complete UTC weeks",
        "decision_rule": "PASS only if Fobs > 2*Q0.9875(Fnull) for all four assets",
        "results": rows,
        "all_assets_memory_majority_gate": bool(all(r["memory_majority_gate"] for r in rows)),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()


