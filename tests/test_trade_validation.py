from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.trade_validation import (
    aggregate_trade_hours,
    normalization_calibration,
    read_daily_trades,
    reconcile_hourly,
)


def _archive(path: Path) -> Path:
    rows = [
        "trade_id,price,qty,quote_qty,time,is_buyer_maker,is_best_match",
        "1,100,1,100,1738540800000000,false,true",
        "2,100,3,300,1738542600000000,true,true",
        "3,100,2,200,1738544400000000,false,true",
        "4,100,4,400,1738548000000000,true,true",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trades.csv", "\n".join(rows))
    return path


def test_trade_parser_and_hourly_reconstruction(tmp_path):
    trades = read_daily_trades(_archive(tmp_path / "sample.zip"))
    assert len(trades) == 4
    assert str(trades.index.tz) == "UTC"
    hours = aggregate_trade_hours(trades)
    assert np.isclose(hours.iloc[0]["Volume"], 4.0)
    assert np.isclose(hours.iloc[0]["tbBase"], 1.0)
    assert int(hours.iloc[0]["trades"]) == 2
    assert np.isclose(hours.iloc[0]["signed_flow"], -2.0)


def test_calibration_and_reconciliation(tmp_path):
    trades = read_daily_trades(_archive(tmp_path / "sample.zip"))
    hours = aggregate_trade_hours(trades)
    calibration = normalization_calibration(trades)
    assert np.isclose(calibration["median_trade_size_exact"], 2.5)
    assert np.isclose(calibration["proxy_to_exact_ratio"], 0.8)
    comparison = reconcile_hourly(hours, hours)
    assert comparison["Volume_max_relative_error"] == 0.0
    assert comparison["trades_sum_relative_error"] == 0.0

