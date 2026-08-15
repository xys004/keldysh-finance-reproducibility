"""Robust controls for the counting-noise claims.

The absolute Fano normalization used by the exploratory analysis is sensitive
to the trade-size distribution.  This module therefore separates two questions:

1. how large is the reported Fano factor under a declared transfer quantum; and
2. how much of the finite-window variance is caused by inter-candle memory.

The second question is answered by stratified permutation ratios.  We report
raw signed base volume and volume-normalised imbalance in parallel.  Three
non-additive nulls probe complete signed-flow ordering, sign ordering, and
magnitude ordering.  An exact finite-sample decomposition separately attributes
the excess to sign order at homogenised magnitudes and a sign--size remainder.
All variance ratios are independent of any proxy transfer quantum.
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


def _sign_size_weekly_components(
    prepared: PreparedHourlyFlow,
    stratification: str,
) -> dict[str, np.ndarray | float]:
    """Return exact weekly pieces of the signed-flow second moment.

    The observed, stratum-demeaned flow is written ``x_t = s_t m_t``.  Its
    diagonal contribution is retained exactly.  To isolate temporal sign
    organisation on the same calendar footing as the permutation null, signs
    are centred within each stratum and magnitudes are replaced by their
    stratum means.  The difference between the observed and homogenised
    off-diagonal terms is the declared sign--size coupling remainder.

    Every complete week contains at most one observation from a given
    ``(period, hour-of-week)`` stratum.  Consequently the diagonal term is also
    the exact expectation of the joint-order permutation variance (up to the
    common sample-variance divisor), rather than a Monte Carlo estimate.
    """
    x = _demean_within_strata(prepared, stratification)
    weeks = int(prepared.n_complete_weeks)
    if len(x) != weeks * HOURS_PER_WEEK:
        raise ValueError("prepared arrays are not complete weekly blocks")

    period = _stratum_period(prepared, stratification)
    labels = pd.MultiIndex.from_arrays([period, prepared.hour_of_week])
    signs = np.sign(x)
    magnitudes = np.abs(x)
    mean_sign = (
        pd.Series(signs, index=labels)
        .groupby(level=[0, 1])
        .transform("mean")
        .to_numpy(float)
    )
    mean_magnitude = (
        pd.Series(magnitudes, index=labels)
        .groupby(level=[0, 1])
        .transform("mean")
        .to_numpy(float)
    )
    sign_only = (signs - mean_sign) * mean_magnitude

    x_week = x.reshape(weeks, HOURS_PER_WEEK)
    sign_week = sign_only.reshape(weeks, HOURS_PER_WEEK)
    weekly_flow = x_week.sum(axis=1)
    weekly_sign_flow = sign_week.sum(axis=1)
    weekly_diagonal = np.square(x_week).sum(axis=1)
    weekly_sign_diagonal = np.square(sign_week).sum(axis=1)
    weekly_total_offdiagonal = np.square(weekly_flow) - weekly_diagonal
    weekly_sign_offdiagonal = (
        np.square(weekly_sign_flow) - weekly_sign_diagonal
    )
    weekly_coupling = weekly_total_offdiagonal - weekly_sign_offdiagonal

    reconstruction = (
        weekly_diagonal + weekly_sign_offdiagonal + weekly_coupling
    )
    error = float(np.max(np.abs(reconstruction - np.square(weekly_flow))))
    scale = float(max(1.0, np.max(np.square(weekly_flow))))
    if error > 5e-12 * scale:
        raise AssertionError("sign--size components do not reconstruct Q_w^2")

    return {
        "weekly_flow": weekly_flow,
        "weekly_sign_flow": weekly_sign_flow,
        "weekly_diagonal": weekly_diagonal,
        "weekly_total_offdiagonal": weekly_total_offdiagonal,
        "weekly_sign_offdiagonal": weekly_sign_offdiagonal,
        "weekly_coupling": weekly_coupling,
        "max_weekly_reconstruction_error": error,
        "max_weekly_reconstruction_relative_error": error / scale,
        "max_stratum_flow_mean": float(
            pd.Series(x, index=labels)
            .groupby(level=[0, 1])
            .mean()
            .abs()
            .max()
        ),
        "max_stratum_sign_proxy_mean": float(
            pd.Series(sign_only, index=labels)
            .groupby(level=[0, 1])
            .mean()
            .abs()
            .max()
        ),
    }


def _bootstrap_sign_size_components(
    components: dict[str, np.ndarray | float],
    block_lengths: tuple[int, ...],
    replicates: int,
    seed: int,
) -> list[dict]:
    """Paired circular-block uncertainty for exact attribution shares."""
    q = np.asarray(components["weekly_flow"], float)
    q_sign = np.asarray(components["weekly_sign_flow"], float)
    diagonal = np.asarray(components["weekly_diagonal"], float)
    total_offdiagonal = np.asarray(
        components["weekly_total_offdiagonal"], float
    )
    sign_offdiagonal = np.asarray(
        components["weekly_sign_offdiagonal"], float
    )
    n = len(q)
    if replicates < 99:
        raise ValueError("at least 99 bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    rows = []
    for requested in block_lengths:
        block = min(max(1, int(requested)), n)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(replicates, n_blocks))
        offsets = np.arange(block)
        indices = (starts[:, :, None] + offsets[None, None, :]) % n
        indices = indices.reshape(replicates, -1)[:, :n]

        q_b = q[indices]
        q_sign_b = q_sign[indices]
        diagonal_sum = diagonal[indices].sum(axis=1)
        total_numerator = (
            total_offdiagonal[indices].sum(axis=1)
            - n * np.square(q_b.mean(axis=1))
        )
        sign_numerator = (
            sign_offdiagonal[indices].sum(axis=1)
            - n * np.square(q_sign_b.mean(axis=1))
        )
        coupling_numerator = total_numerator - sign_numerator
        valid = total_numerator > 0.0
        if np.count_nonzero(valid) < max(50, replicates // 2):
            raise ValueError("too few positive-excess bootstrap replicates")
        sign_share = sign_numerator[valid] / total_numerator[valid]
        coupling_share = coupling_numerator[valid] / total_numerator[valid]
        amplification = 1.0 + total_numerator / diagonal_sum
        rows.append({
            "block_length_weeks": block,
            "replicates": int(replicates),
            "positive_excess_replicates": int(np.count_nonzero(valid)),
            "positive_excess_fraction": float(np.mean(valid)),
            "amplification_median": float(np.median(amplification)),
            "amplification_ci025": float(np.quantile(amplification, 0.025)),
            "amplification_ci975": float(np.quantile(amplification, 0.975)),
            "sign_share_median": float(np.median(sign_share)),
            "sign_share_ci025": float(np.quantile(sign_share, 0.025)),
            "sign_share_ci975": float(np.quantile(sign_share, 0.975)),
            "coupling_share_median": float(np.median(coupling_share)),
            "coupling_share_ci025": float(np.quantile(coupling_share, 0.025)),
            "coupling_share_ci975": float(np.quantile(coupling_share, 0.975)),
            "probability_coupling_share_gt_half": float(
                np.mean(coupling_share > 0.5)
            ),
            "probability_sign_component_positive": float(
                np.mean(sign_numerator > 0.0)
            ),
        })
    return rows


def stratified_sign_size_decomposition(
    prepared: PreparedHourlyFlow,
    stratification: str = "quarterly",
    bootstrap_replicates: int = 0,
    bootstrap_block_lengths: tuple[int, ...] = (4, 8, 13, 26),
    seed: int = 20260815,
) -> dict:
    """Decompose weekly variance excess into sign and sign--size terms.

    The reference is the exact expectation of the joint-order permutation
    variance.  The sign term keeps the observed, stratum-centred sign path but
    homogenises magnitudes within each stratum.  The coupling term is the exact
    remainder.  This is a declared fixed-diagonal, non-orthogonal
    finite-sample decomposition, not a unique causal allocation of an
    interaction.  In particular, the reference plus sign component is not
    the ordinary variance of the magnitude-homogenised sign proxy.
    """
    components = _sign_size_weekly_components(prepared, stratification)
    q = np.asarray(components["weekly_flow"], float)
    diagonal = np.asarray(components["weekly_diagonal"], float)
    total_offdiagonal = np.asarray(
        components["weekly_total_offdiagonal"], float
    )
    sign_offdiagonal = np.asarray(
        components["weekly_sign_offdiagonal"], float
    )
    weeks = len(q)
    divisor = weeks - 1
    observed = float(np.var(q, ddof=1))
    marginal_reference = float(diagonal.sum() / divisor)
    sign_component = float(sign_offdiagonal.sum() / divisor)
    total_excess = observed - marginal_reference
    coupling_component = total_excess - sign_component
    reconstruction = marginal_reference + sign_component + coupling_component
    if not np.isclose(reconstruction, observed, rtol=5e-13, atol=5e-13):
        raise AssertionError("variance decomposition is not exact")
    if total_excess <= 0:
        raise ValueError("observed weekly variance has no positive memory excess")

    result = {
        "flow_mode": prepared.flow_mode,
        "stratification": stratification,
        "horizon_hours": HOURS_PER_WEEK,
        "n_complete_weeks": int(weeks),
        "n_candles": int(len(prepared.index)),
        "variance_observed": observed,
        "marginal_reference": marginal_reference,
        "amplification_over_marginal": float(observed / marginal_reference),
        "total_excess": total_excess,
        "sign_order_component": sign_component,
        "sign_size_coupling_component": coupling_component,
        "sign_share_of_excess": float(sign_component / total_excess),
        "coupling_share_of_excess": float(coupling_component / total_excess),
        "weekly_flow_mean": float(q.mean()),
        "centering_correction": float(-weeks * np.square(q.mean()) / divisor),
        "exact_reconstruction_error": float(abs(reconstruction - observed)),
        "max_weekly_reconstruction_error": components[
            "max_weekly_reconstruction_error"
        ],
        "max_weekly_reconstruction_relative_error": components[
            "max_weekly_reconstruction_relative_error"
        ],
        "max_stratum_flow_mean": components["max_stratum_flow_mean"],
        "max_stratum_sign_proxy_mean": components[
            "max_stratum_sign_proxy_mean"
        ],
        "interpretation": (
            "fixed-diagonal, non-orthogonal finite-sample attribution "
            "relative to the analytic joint-order expectation; the "
            "interaction share is conditional on within-stratum sign "
            "centering and arithmetic-mean magnitude homogenisation and is "
            "not a unique causal allocation"
        ),
    }
    if bootstrap_replicates:
        result["paired_weekly_block_bootstrap"] = _bootstrap_sign_size_components(
            components,
            bootstrap_block_lengths,
            bootstrap_replicates,
            seed,
        )
    return result


def paired_normalization_share_difference(
    prepared_raw: PreparedHourlyFlow,
    prepared_normalized: PreparedHourlyFlow,
    stratification: str = "quarterly",
    bootstrap_replicates: int = 3999,
    bootstrap_block_lengths: tuple[int, ...] = (4, 8, 13, 26),
    seed: int = 20260815,
) -> dict:
    """Estimate the paired change in sign share after volume normalisation."""
    if prepared_raw.flow_mode != "raw":
        raise ValueError("prepared_raw must use raw flow")
    if prepared_normalized.flow_mode != "normalized":
        raise ValueError("prepared_normalized must use normalized flow")
    if not prepared_raw.index.equals(prepared_normalized.index):
        raise ValueError("raw and normalized flows must use identical timestamps")
    if bootstrap_replicates < 99:
        raise ValueError("at least 99 bootstrap replicates are required")

    raw = _sign_size_weekly_components(prepared_raw, stratification)
    normalized = _sign_size_weekly_components(prepared_normalized, stratification)
    n = len(np.asarray(raw["weekly_flow"]))
    rng = np.random.default_rng(seed)

    def share(c: dict[str, np.ndarray | float], indices: np.ndarray) -> np.ndarray:
        q = np.asarray(c["weekly_flow"], float)[indices]
        q_sign = np.asarray(c["weekly_sign_flow"], float)[indices]
        total = (
            np.asarray(c["weekly_total_offdiagonal"], float)[indices].sum(axis=1)
            - n * np.square(q.mean(axis=1))
        )
        sign = (
            np.asarray(c["weekly_sign_offdiagonal"], float)[indices].sum(axis=1)
            - n * np.square(q_sign.mean(axis=1))
        )
        return np.divide(
            sign,
            total,
            out=np.full_like(sign, np.nan),
            where=total > 0.0,
        )

    point_index = np.arange(n, dtype=int)[None, :]
    point_raw = float(share(raw, point_index)[0])
    point_normalized = float(share(normalized, point_index)[0])
    rows = []
    for requested in bootstrap_block_lengths:
        block = min(max(1, int(requested)), n)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(bootstrap_replicates, n_blocks))
        offsets = np.arange(block)
        indices = (starts[:, :, None] + offsets[None, None, :]) % n
        indices = indices.reshape(bootstrap_replicates, -1)[:, :n]
        raw_share = share(raw, indices)
        normalized_share = share(normalized, indices)
        valid = np.isfinite(raw_share) & np.isfinite(normalized_share)
        delta = normalized_share[valid] - raw_share[valid]
        if len(delta) < max(50, bootstrap_replicates // 2):
            raise ValueError("too few paired positive-excess bootstrap replicates")
        rows.append({
            "block_length_weeks": block,
            "replicates": int(bootstrap_replicates),
            "valid_paired_replicates": int(len(delta)),
            "delta_sign_share_median": float(np.median(delta)),
            "delta_sign_share_ci025": float(np.quantile(delta, 0.025)),
            "delta_sign_share_ci975": float(np.quantile(delta, 0.975)),
            "probability_delta_sign_share_positive": float(np.mean(delta > 0.0)),
        })
    return {
        "stratification": stratification,
        "n_complete_weeks": int(n),
        "raw_sign_share": point_raw,
        "normalized_sign_share": point_normalized,
        "delta_sign_share": point_normalized - point_raw,
        "paired_weekly_block_bootstrap": rows,
    }


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
        "variance_null_mean": float(np.mean(null_variance)),
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
    if null_mode == "joint":
        analytic = float(np.square(j).sum() / (weeks - 1))
        result.update({
            "variance_null_analytic_expectation": analytic,
            "null_mean_relative_error_to_analytic": float(
                (np.mean(null_variance) - analytic) / analytic
            ),
        })
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
