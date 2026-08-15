"""Trade-level calibration of the hourly counting-noise normalization."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TRADE_COLUMNS = (
    "trade_id", "price", "qty", "quote_qty", "time",
    "is_buyer_maker", "is_best_match",
)


def download_daily_trades(symbol: str, date: str, cache_dir: str | Path) -> Path:
    """Download one public Binance Vision spot-trades archive, with caching."""
    import requests

    symbol = symbol.upper().replace("-", "")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}-trades-{date}.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = (
        "https://data.binance.vision/data/spot/daily/trades/"
        f"{symbol}/{symbol}-trades-{date}.zip"
    )
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    temporary = path.with_suffix(".zip.part")
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temporary.replace(path)
    return path


def read_daily_trades(path: str | Path) -> pd.DataFrame:
    """Read and validate a Binance Vision ``trades`` archive.

    The public archive changed from millisecond to microsecond timestamps in
    2025. The unit is inferred from the integer magnitude and recorded in a
    UTC index. Header and headerless CSV variants are both accepted.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV member in {path}, found {len(members)}")
        raw = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(raw), header=None, names=TRADE_COLUMNS, low_memory=False)
    for column in ("trade_id", "price", "qty", "quote_qty", "time"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_id", "price", "qty", "quote_qty", "time"])
    if frame.empty:
        raise ValueError(f"no numeric trades in {path}")
    if frame["trade_id"].duplicated().any():
        raise ValueError("duplicate trade identifiers")
    numeric = frame[["price", "qty", "quote_qty", "time"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite trade fields")
    if np.any(frame["price"].to_numpy(float) <= 0) or np.any(frame["qty"].to_numpy(float) <= 0):
        raise ValueError("prices and base quantities must be positive")
    quote_expected = frame["price"].to_numpy(float) * frame["qty"].to_numpy(float)
    quote_reported = frame["quote_qty"].to_numpy(float)
    if not np.allclose(quote_reported, quote_expected, rtol=2e-6, atol=2e-8):
        raise ValueError("quote quantities are incompatible with price times base quantity")

    maker = frame["is_buyer_maker"].astype(str).str.lower().map({"true": True, "false": False})
    if maker.isna().any():
        raise ValueError("invalid is_buyer_maker values")
    frame["is_buyer_maker"] = maker.to_numpy(bool)
    stamps = frame["time"].to_numpy(np.int64)
    unit = "us" if int(np.median(stamps)) >= 100_000_000_000_000 else "ms"
    frame.index = pd.to_datetime(stamps, unit=unit, utc=True)
    frame.index.name = "Datetime"
    return frame.sort_index()


def aggregate_trade_hours(trades: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct hourly base volume, taker-buy volume, counts, and flow."""
    qty = trades["qty"].to_numpy(float)
    buyer_taker = ~trades["is_buyer_maker"].to_numpy(bool)
    work = pd.DataFrame(
        {
            "Volume": qty,
            "tbBase": np.where(buyer_taker, qty, 0.0),
            "trades": np.ones(len(trades), dtype=np.int64),
        },
        index=trades.index,
    )
    hourly = work.resample("1h").sum()
    hourly["signed_flow"] = 2.0 * hourly["tbBase"] - hourly["Volume"]
    return hourly


def normalization_calibration(trades: pd.DataFrame) -> dict:
    """Compare exact median trade size with the hourly mean-size proxy."""
    hourly = aggregate_trade_hours(trades)
    exact = float(np.median(trades["qty"].to_numpy(float)))
    active = hourly["trades"].to_numpy(float) > 0
    proxy = float(np.median(
        hourly.loc[active, "Volume"].to_numpy(float)
        / hourly.loc[active, "trades"].to_numpy(float)
    ))
    return {
        "n_trades": int(len(trades)),
        "n_hours": int(len(hourly)),
        "median_trade_size_exact": exact,
        "median_hourly_mean_size_proxy": proxy,
        "proxy_to_exact_ratio": float(proxy / exact),
        "fano_exact_to_proxy_multiplier": float((proxy / exact) ** 2),
    }


def reconcile_hourly(trade_hours: pd.DataFrame, klines: pd.DataFrame) -> dict:
    """Reconcile trade-derived hourly fields against public kline fields."""
    common = trade_hours.index.intersection(klines.index)
    if len(common) != len(trade_hours):
        raise ValueError("kline comparison does not cover every reconstructed hour")
    errors: dict[str, float | int] = {"n_hours_compared": int(len(common))}
    for column in ("Volume", "tbBase", "trades"):
        actual = trade_hours.loc[common, column].to_numpy(float)
        reference = klines.loc[common, column].to_numpy(float)
        scale = np.maximum(np.abs(reference), 1.0 if column == "trades" else 1e-12)
        errors[f"{column}_max_relative_error"] = float(np.max(np.abs(actual - reference) / scale))
        errors[f"{column}_sum_relative_error"] = float(
            abs(actual.sum() - reference.sum()) / max(abs(reference.sum()), 1e-12)
        )
    return errors
