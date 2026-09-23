"""Calibrate the Fano normalization with public individual trades."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.flow import fetch_klines_with_flow
from keldysh_finance.trade_validation import (
    aggregate_trade_hours,
    download_daily_trades,
    normalization_calibration,
    read_daily_trades,
    reconcile_hourly,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp13_trade_level_validation.json"
DATE = "2025-03-03"
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def main() -> None:
    rows = []
    for symbol in SYMBOLS:
        archive = download_daily_trades(symbol, DATE, ROOT / "output" / "cache" / "trades")
        trades = read_daily_trades(archive)
        hourly = aggregate_trade_hours(trades)
        klines = fetch_klines_with_flow(symbol, interval="1h", years=4.0)
        row = {
            "symbol": symbol,
            "date_utc": DATE,
            **normalization_calibration(trades),
            **reconcile_hourly(hourly, klines),
        }
        rows.append(row)
        print(
            f"{symbol}: n={row['n_trades']:,}, proxy/exact="
            f"{row['proxy_to_exact_ratio']:.3f}, F multiplier="
            f"{row['fano_exact_to_proxy_multiplier']:.2f}",
            flush=True,
        )
    payload = {
        "source": "Binance Vision public spot daily trades",
        "purpose": "normalization calibration, not an independent long-memory sample",
        "results": rows,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
