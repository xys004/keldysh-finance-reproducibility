"""Robust controls for the counting-noise claims.

The absolute Fano normalization used by the exploratory analysis is sensitive
to the trade-size distribution.  This module therefore separates two questions:

1. how large is the reported Fano factor under a declared transfer quantum; and
2. how much of the finite-window variance is caused by inter-candle memory.

The second question is answered by stratified permutation ratios.  We report
raw signed base volume and volume-normalised imbalance in parallel.  Three
nulls separate the effects of the complete signed-flow ordering, sign memory,
and magnitude clustering.  The variance ratios are independent of any proxy
transfer quantum.
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
    flow_mode: str


STRATIFICATIONS = ("quarterly", "monthly", "biweekly")
FLOW_MODES = ("raw", "normalized")
NULL_MODES = ("joint", "sign", "magnitude")


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


def prepare_complete_weeks(
    df: pd.DataFrame,
    flow_mode: str = "raw",
) -> PreparedHourlyFlow:
    """Return complete Monday--Sunday UTC weeks for a declared flow observable.

    ``raw`` is signed base volume, ``2*tbBase-Volume``.  ``normalized`` is the
    dimensionless signed imbalance obtained by dividing that quantity by
    volume.  Zero-volume candles have zero normalised imbalance.
    """
    if flow_mode not in FLOW_MODES:
        raise ValueError(f"flow_mode must be one of {FLOW_MODES}, got {flow_mode!r}")
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
    raw_flow = 2.0 * x["tbBase"].to_numpy(float) - x["Volume"].to_numpy(float)
    trades = x["trades"].to_numpy(float)
    volume = x["Volume"].to_numpy(float)
    if flow_mode == "raw":
        flow = raw_flow
    else:
        flow = np.divide(
            raw_flow,
            volume,
            out=np.zeros_like(raw_flow),
            where=volume > 0,
        )

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
        flow_mode=flow_mode,
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


def _circular_block_bootstrap_variance_ratio(
    weekly_values: np.ndarray,
    denominator: float,
    block_lengths: tuple[int, ...],
    replicates: int,
    seed: int,
) -> list[dict]:
    """Conditional moving-block bootstrap for a weekly variance ratio."""
    values = np.asarray(weekly_values, float)
    n = len(values)
    if replicates < 99:
        raise ValueError("at least 99 bootstrap replicates are required")
    if n < 8 or denominator <= 0:
        raise ValueError("invalid weekly series or null denominator")
    rng = np.random.default_rng(seed)
    rows = []
    for requested in block_lengths:
        block = min(max(1, int(requested)), n)
        n_blocks = int(np.ceil(n / block))
        start = rng.integers(0, n, size=(replicates, n_blocks))
        offsets = np.arange(block)
        indices = (start[:, :, None] + offsets[None, None, :]) % n
        samples = values[indices.reshape(replicates, -1)[:, :n]]
        ratios = samples.var(axis=1, ddof=1) / denominator
        rows.append({
            "block_length_weeks": block,
            "replicates": int(replicates),
            "ratio_median": float(np.median(ratios)),
            "ratio_ci025": float(np.quantile(ratios, 0.025)),
            "ratio_ci975": float(np.quantile(ratios, 0.975)),
            "probability_ratio_gt_1": float(np.mean(ratios > 1.0)),
            "probability_ratio_gt_2": float(np.mean(ratios > 2.0)),
        })
    return rows


def stratified_variance_memory_test(
    prepared: PreparedHourlyFlow,
    permutations: int = 9999,
    seed: int = 20260814,
    batch_size: int = 128,
    null_quantile: float = 0.9875,
    stratification: str = "quarterly",
    null_mode: str = "joint",
    bootstrap_replicates: int = 0,
    bootstrap_block_lengths: tuple[int, ...] = (4, 8, 13, 26),
) -> dict:
    """Test weekly variance amplification against a declared stratified null.

    ``joint`` permutes the complete demeaned signed-flow values.  ``sign``
    preserves the time ordering of absolute magnitudes and permutes signs.
    ``magnitude`` preserves the sign sequence and permutes absolute magnitudes.
    All permutations are within ``(period, hour-of-week)`` strata and are
    re-centred within those strata.  Sign and magnitude nulls deliberately do
    not preserve contemporaneous sign--magnitude pairing.
    """
    if permutations < 99:
        raise ValueError("at least 99 permutations are required")
    if not (0.5 < null_quantile < 1.0):
        raise ValueError("null_quantile must lie between 0.5 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if null_mode not in NULL_MODES:
        raise ValueError(f"null_mode must be one of {NULL_MODES}, got {null_mode!r}")

    j = _demean_within_strata(prepared, stratification)
    n = np.asarray(prepared.trades, float)
    v = np.asarray(prepared.volume, float)
    weeks = int(prepared.n_complete_weeks)
    if len(j) != weeks * HOURS_PER_WEEK:
        raise ValueError("prepared arrays are not complete weekly blocks")
    weekly_j = j.reshape(weeks, HOURS_PER_WEEK).sum(axis=1)
    weekly_n = n.reshape(weeks, HOURS_PER_WEEK).sum(axis=1)
    mean_n = float(np.mean(weekly_n))
    var_obs = float(np.var(weekly_j, ddof=1))
    qbar_proxy = None
    f_obs = None
    if prepared.flow_mode == "raw":
        valid_q = n > 0
        qbar_proxy = float(np.median(v[valid_q] / n[valid_q]))
        if not np.isfinite(qbar_proxy) or qbar_proxy <= 0:
            raise ValueError("invalid proxy transfer quantum")
        f_obs = var_obs / (qbar_proxy ** 2 * mean_n)

    groups, singleton_fraction, min_size, max_size = _stratum_positions(
        prepared, stratification
    )
    groups_by_size: dict[int, np.ndarray] = {}
    for size in sorted({len(pos) for pos in groups}):
        groups_by_size[size] = np.stack([pos for pos in groups if len(pos) == size])
    rng = np.random.default_rng(seed)
    null_variance = np.empty(permutations, dtype=float)
    cursor = 0
    while cursor < permutations:
        b = min(batch_size, permutations - cursor)
        jp = np.empty((b, len(j)), dtype=float)
        for size, positions in groups_by_size.items():
            order = np.argsort(
                rng.random((b, len(positions), size)), axis=2
            )
            values = j[positions][None, :, :]
            flat_positions = positions.reshape(-1)
            if null_mode == "joint":
                permuted = np.take_along_axis(values, order, axis=2)
            elif null_mode == "sign":
                signs = np.sign(values)
                permuted = np.abs(values) * np.take_along_axis(signs, order, axis=2)
            else:
                magnitudes = np.abs(values)
                permuted = np.sign(values) * np.take_along_axis(
                    magnitudes, order, axis=2
                )
            permuted = permuted - permuted.mean(axis=2, keepdims=True)
            jp[:, flat_positions] = permuted.reshape(b, -1)
        weekly_jp = jp.reshape(b, weeks, HOURS_PER_WEEK).sum(axis=2)
        null_variance[cursor:cursor + b] = weekly_jp.var(axis=1, ddof=1)
        cursor += b

    q_null = float(np.quantile(null_variance, null_quantile))
    med_null = float(np.median(null_variance))
    empirical_p = float(
        (1 + np.count_nonzero(null_variance >= var_obs)) / (permutations + 1)
    )
    result = {
        "flow_mode": prepared.flow_mode,
        "null_mode": null_mode,
        "stratification": stratification,
        "horizon_hours": HOURS_PER_WEEK,
        "n_complete_weeks": weeks,
        "n_candles": int(len(j)),
        "permutations": int(permutations),
        "null_quantile_level": float(null_quantile),
        "variance_observed": var_obs,
        "variance_null_median": med_null,
        "variance_null_quantile": q_null,
        "variance_ratio_to_null_median": float(var_obs / med_null),
        "variance_ratio_to_null_quantile": float(var_obs / q_null),
        "above_twice_null_median": bool(var_obs > 2.0 * med_null),
        "above_null_quantile": bool(var_obs > q_null),
        "memory_majority_gate": bool(var_obs > 2.0 * q_null),
        "empirical_p_upper": empirical_p,
        "n_strata": int(len(groups)),
        "min_stratum_size": min_size,
        "max_stratum_size": max_size,
        "singleton_fraction": singleton_fraction,
        "null_variance_q025": float(np.quantile(null_variance, 0.025)),
        "null_variance_q975": float(np.quantile(null_variance, 0.975)),
        "sign_magnitude_pairing_preserved": bool(null_mode == "joint"),
    }
    if prepared.flow_mode == "raw":
        scale = qbar_proxy ** 2 * mean_n
        result.update({
            "qbar_proxy": qbar_proxy,
            "fano_observed": float(f_obs),
            "fano_null_median": float(med_null / scale),
            "fano_null_quantile": float(q_null / scale),
            "null_fano_q025": float(np.quantile(null_variance, 0.025) / scale),
            "null_fano_q975": float(np.quantile(null_variance, 0.975) / scale),
        })
    if bootstrap_replicates:
        result["weekly_block_bootstrap"] = _circular_block_bootstrap_variance_ratio(
            weekly_j,
            med_null,
            bootstrap_block_lengths,
            bootstrap_replicates,
            seed + 1_000_003,
        )
    return result


def stratified_fano_memory_test(
    prepared: PreparedHourlyFlow,
    permutations: int = 9999,
    seed: int = 20260814,
    batch_size: int = 128,
    null_quantile: float = 0.9875,
    stratification: str = "quarterly",
) -> dict:
    """Backward-compatible raw-flow wrapper for the joint permutation null."""
    if prepared.flow_mode != "raw":
        raise ValueError("Fano normalisation is defined only for raw signed volume")
    return stratified_variance_memory_test(
        prepared,
        permutations=permutations,
        seed=seed,
        batch_size=batch_size,
        null_quantile=null_quantile,
        stratification=stratification,
        null_mode="joint",
    )
