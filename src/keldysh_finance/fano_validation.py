"""Robust controls for the counting-noise claims.

The absolute Fano normalization used by the exploratory analysis is sensitive
to the trade-size distribution.  This module therefore separates two questions:

1. how large is the reported Fano factor under a declared transfer quantum; and
2. how much of the finite-window variance is caused by inter-candle memory.

The second question is answered by a stratified permutation ratio.  The null
keeps the empirical signed-flow marginal, heavy tails, activity, seasonality,
quarterly scale changes, and the flow/activity pairing.  It destroys only the
ordering across candles.  The ratio is independent of the transfer quantum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


HOURS_PER_WEEK = 168


@dataclass(frozen=True)
class PreparedHourlyFlow:
    index: pd.DatetimeIndex
    flow: np.ndarray
    flow_demeaned: np.ndarray
    volume: np.ndarray
    trades: np.ndarray
    quarter: np.ndarray
    hour_of_week: np.ndarray
    week_start: pd.DatetimeIndex
    n_complete_weeks: int


STRATIFICATIONS = ("quarterly", "monthly", "biweekly")


def validate_hourly_klines(df: pd.DataFrame) -> pd.DataFrame:
    """Validate base-unit Binance hourly fields without silently repairing them."""
    required = {"Volume", "tbBase", "trades"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing hourly fields: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("hourly frame must use a DatetimeIndex")

    out = df.loc[:, ["Volume", "tbBase", "trades"]].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        raise ValueError("hourly timestamps must be timezone-aware UTC")
    idx = idx.tz_convert("UTC")
    if idx.has_duplicates:
        raise ValueError("duplicate hourly timestamps")
    if not idx.is_monotonic_increasing:
        raise ValueError("hourly timestamps must be sorted")
    if np.any((idx.minute != 0) | (idx.second != 0) | (idx.microsecond != 0)):
        raise ValueError("timestamps must lie on exact UTC hours")
    out.index = idx

    x = out.to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("non-finite Volume/tbBase/trades values")
    volume = out["Volume"].to_numpy(float)
    buy = out["tbBase"].to_numpy(float)
    trades = out["trades"].to_numpy(float)
    tol = 1e-10 * np.maximum(1.0, np.abs(volume))
    if np.any(volume < -tol):
        raise ValueError("negative base volume")
    if np.any(buy < -tol) or np.any(buy > volume + tol):
        raise ValueError("taker-buy base volume must satisfy 0 <= tbBase <= Volume")
    if np.any(trades < 0) or np.any(np.abs(trades - np.rint(trades)) > 1e-8):
        raise ValueError("trade counts must be non-negative integers")
    if np.any((trades == 0) & (np.abs(volume) > tol)):
        raise ValueError("positive volume with zero trades")
    return out


def prepare_complete_weeks(df: pd.DataFrame) -> PreparedHourlyFlow:
    """Return complete Monday-Sunday UTC weeks with a stratified mean removed."""
    x = validate_hourly_klines(df)
    idx = x.index
    week_start = idx.normalize() - pd.to_timedelta(idx.dayofweek, unit="D")

    keep = np.zeros(len(x), dtype=bool)
    for _, pos in pd.Series(np.arange(len(x)), index=week_start).groupby(level=0):
        p = pos.to_numpy(int)
        if len(p) != HOURS_PER_WEEK:
            continue
        expected = pd.date_range(idx[p[0]], periods=HOURS_PER_WEEK, freq="h", tz="UTC")
        if idx[p[0]].dayofweek != 0 or idx[p[0]].hour != 0:
            continue
        if idx[p].equals(expected):
            keep[p] = True
    if keep.sum() < 8 * HOURS_PER_WEEK:
        raise ValueError("fewer than eight complete UTC weeks")

    x = x.iloc[np.flatnonzero(keep)]
    idx = x.index
    week_start = idx.normalize() - pd.to_timedelta(idx.dayofweek, unit="D")
    quarter = (idx.year.to_numpy(int) * 4
               + ((idx.month.to_numpy(int) - 1) // 3))
    hour_of_week = idx.dayofweek.to_numpy(int) * 24 + idx.hour.to_numpy(int)
    flow = 2.0 * x["tbBase"].to_numpy(float) - x["Volume"].to_numpy(float)
    trades = x["trades"].to_numpy(float)
    volume = x["Volume"].to_numpy(float)

    labels = pd.MultiIndex.from_arrays([quarter, hour_of_week])
    counts = pd.Series(np.ones(len(x), dtype=int), index=labels).groupby(level=[0, 1]).sum()
    if int(counts.min()) < 2:
        raise ValueError("every quarter-hour-of-week stratum must contain >=2 candles")
    means = pd.Series(flow, index=labels).groupby(level=[0, 1]).transform("mean").to_numpy()
    demeaned = flow - means
    n_weeks = len(x) // HOURS_PER_WEEK
    if len(x) != n_weeks * HOURS_PER_WEEK:
        raise AssertionError("complete-week preparation lost rectangularity")
    return PreparedHourlyFlow(
        index=idx,
        flow=flow,
        flow_demeaned=demeaned,
        volume=volume,
        trades=trades,
        quarter=quarter,
        hour_of_week=hour_of_week,
        week_start=pd.DatetimeIndex(week_start),
        n_complete_weeks=n_weeks,
    )


def _stratum_period(prepared: PreparedHourlyFlow, stratification: str) -> np.ndarray:
    """Calendar/block label used together with hour of week.

    ``biweekly`` uses consecutive 14-day blocks anchored at the first complete
    Monday in the retained sample.  Calendar month and quarter strata can have
    singleton edge cells after incomplete weeks are removed; these observations
    are retained but remain fixed under permutation and are reported explicitly.
    """
    if stratification not in STRATIFICATIONS:
        raise ValueError(
            f"stratification must be one of {STRATIFICATIONS}, got {stratification!r}"
        )
    idx = prepared.index
    if stratification == "quarterly":
        return prepared.quarter
    if stratification == "monthly":
        return idx.year.to_numpy(int) * 12 + idx.month.to_numpy(int) - 1
    elapsed_days = (prepared.week_start - prepared.week_start[0]).days.to_numpy(int)
    return elapsed_days // 14


def _stratum_positions(
    prepared: PreparedHourlyFlow,
    stratification: str,
) -> tuple[list[np.ndarray], float, int, int]:
    period = _stratum_period(prepared, stratification)
    labels = pd.MultiIndex.from_arrays([period, prepared.hour_of_week])
    positions = pd.Series(np.arange(len(labels)), index=labels)
    groups = [g.to_numpy(int) for _, g in positions.groupby(level=[0, 1], sort=False)]
    if not groups:
        raise ValueError("no strata")
    singleton_count = int(sum(len(g) for g in groups if len(g) == 1))
    permutable = [g for g in groups if len(g) > 1]
    if not permutable:
        raise ValueError("no permutable strata")
    return (
        groups,
        float(singleton_count / len(labels)),
        int(min(map(len, groups))),
        int(max(map(len, groups))),
    )


def _demean_within_strata(
    prepared: PreparedHourlyFlow,
    stratification: str,
) -> np.ndarray:
    period = _stratum_period(prepared, stratification)
    labels = pd.MultiIndex.from_arrays([period, prepared.hour_of_week])
    means = (
        pd.Series(prepared.flow, index=labels)
        .groupby(level=[0, 1])
        .transform("mean")
        .to_numpy(float)
    )
    return prepared.flow - means


def stratified_fano_memory_test(
    prepared: PreparedHourlyFlow,
    permutations: int = 9999,
    seed: int = 20260814,
    batch_size: int = 128,
    null_quantile: float = 0.9875,
    stratification: str = "quarterly",
) -> dict:
    """Test weekly variance amplification against a size-preserving null.

    The joint ``(flow, trades)`` pair is permuted within each
    ``(calendar/block period, hour-of-week)`` stratum.  Weekly windows are
    non-overlapping.
    The reported ``variance_ratio`` equals ``F_obs/F_null`` and is independent
    of the proxy transfer quantum.
    """
    if permutations < 99:
        raise ValueError("at least 99 permutations are required")
    if not (0.5 < null_quantile < 1.0):
        raise ValueError("null_quantile must lie between 0.5 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    j = _demean_within_strata(prepared, stratification)
    n = np.asarray(prepared.trades, float)
    v = np.asarray(prepared.volume, float)
    weeks = int(prepared.n_complete_weeks)
    if len(j) != weeks * HOURS_PER_WEEK:
        raise ValueError("prepared arrays are not complete weekly blocks")
    valid_q = n > 0
    qbar_proxy = float(np.median(v[valid_q] / n[valid_q]))
    if not np.isfinite(qbar_proxy) or qbar_proxy <= 0:
        raise ValueError("invalid proxy transfer quantum")

    weekly_j = j.reshape(weeks, HOURS_PER_WEEK).sum(axis=1)
    weekly_n = n.reshape(weeks, HOURS_PER_WEEK).sum(axis=1)
    mean_n = float(np.mean(weekly_n))
    var_obs = float(np.var(weekly_j, ddof=1))
    f_obs = var_obs / (qbar_proxy ** 2 * mean_n)

    groups, singleton_fraction, min_size, max_size = _stratum_positions(
        prepared, stratification
    )
    groups_by_size: dict[int, np.ndarray] = {}
    for size in sorted({len(pos) for pos in groups}):
        groups_by_size[size] = np.stack([pos for pos in groups if len(pos) == size])
    rng = np.random.default_rng(seed)
    null_f = np.empty(permutations, dtype=float)
    cursor = 0
    while cursor < permutations:
        b = min(batch_size, permutations - cursor)
        jp = np.empty((b, len(j)), dtype=float)
        npair = np.empty((b, len(n)), dtype=float)
        for size, positions in groups_by_size.items():
            order = np.argsort(
                rng.random((b, len(positions), size)), axis=2
            )
            j_values = j[positions][None, :, :]
            n_values = n[positions][None, :, :]
            flat_positions = positions.reshape(-1)
            jp[:, flat_positions] = np.take_along_axis(
                j_values, order, axis=2
            ).reshape(b, -1)
            npair[:, flat_positions] = np.take_along_axis(
                n_values, order, axis=2
            ).reshape(b, -1)
        weekly_jp = jp.reshape(b, weeks, HOURS_PER_WEEK).sum(axis=2)
        weekly_np = npair.reshape(b, weeks, HOURS_PER_WEEK).sum(axis=2)
        mean_np = weekly_np.mean(axis=1)
        if not np.allclose(mean_np, mean_n, rtol=0.0, atol=1e-10 * max(mean_n, 1.0)):
            raise AssertionError("joint permutation changed the global mean trade count")
        null_f[cursor:cursor + b] = weekly_jp.var(axis=1, ddof=1) / (
            qbar_proxy ** 2 * mean_np
        )
        cursor += b

    q_null = float(np.quantile(null_f, null_quantile))
    med_null = float(np.median(null_f))
    empirical_p = float((1 + np.count_nonzero(null_f >= f_obs)) / (permutations + 1))
    return {
        "stratification": stratification,
        "horizon_hours": HOURS_PER_WEEK,
        "n_complete_weeks": weeks,
        "n_candles": int(len(j)),
        "permutations": int(permutations),
        "null_quantile_level": float(null_quantile),
        "qbar_proxy": qbar_proxy,
        "fano_observed": float(f_obs),
        "fano_null_median": med_null,
        "fano_null_quantile": q_null,
        "variance_ratio_to_null_median": float(f_obs / med_null),
        "variance_ratio_to_null_quantile": float(f_obs / q_null),
        "above_twice_null_median": bool(f_obs > 2.0 * med_null),
        "above_null_quantile": bool(f_obs > q_null),
        "memory_majority_gate": bool(f_obs > 2.0 * q_null),
        "empirical_p_upper": empirical_p,
        "n_strata": int(len(groups)),
        "min_stratum_size": min_size,
        "max_stratum_size": max_size,
        "singleton_fraction": singleton_fraction,
        "null_fano_q025": float(np.quantile(null_f, 0.025)),
        "null_fano_q975": float(np.quantile(null_f, 0.975)),
    }
