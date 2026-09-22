"""Exhaustive week-shift diagnostic for sign--magnitude synchrony.

The diagnostic preserves the sign path, the magnitude path, and the
hour-of-week phase.  A nonzero circular displacement by an integer number of
weeks breaks their contemporaneous pairing.  It is deliberately a diagnostic:
even an independent pair of persistent paths has a nonzero cross component.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .fano_validation import HOURS_PER_WEEK, expected_recentered_magnitude_second_moment


@dataclass(frozen=True)
class CrossComponents:
    diagonal: float
    sign: float
    cross: float
    cross_uncorrected: float
    recentering_correction: float
    q2: float


@dataclass(frozen=True)
class _StratumPlan:
    codes: np.ndarray
    counts: np.ndarray
    window: int


def _stratum_plan(strata: np.ndarray, window: int) -> _StratumPlan:
    strata = np.asarray(strata)
    _, codes = np.unique(strata, return_inverse=True)
    counts = np.bincount(codes)
    if strata.size % window:
        raise ValueError("the series must contain an integer number of windows")
    labels = codes.reshape(-1, window)
    if any(np.unique(row).size != window for row in labels):
        raise ValueError("each window must contain at most one position per stratum")
    return _StratumPlan(codes=codes, counts=counts, window=window)


def _recenter(values: np.ndarray, plan: _StratumPlan) -> np.ndarray:
    means = np.bincount(plan.codes, weights=values) / plan.counts
    return values - means[plan.codes]


def recenter_within_strata(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Subtract the mean of ``values`` in each declared stratum."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if values.ndim != 1 or strata.ndim != 1 or values.size != strata.size:
        raise ValueError("values and strata must be aligned one-dimensional arrays")
    return _recenter(values, _stratum_plan(strata, HOURS_PER_WEEK))


def misalign_magnitudes_by_weeks(
    raw_flow: np.ndarray,
    n_complete_weeks: int,
    shift_weeks: int,
) -> np.ndarray:
    """Return ``s_t m_{t+Delta}``, with ``Delta`` an integer week shift.

    ``shift_weeks=0`` is permitted only for verification.  The exhaustive
    surrogate distribution must use every nonzero shift exactly once.
    """
    raw_flow = np.asarray(raw_flow, dtype=float)
    if raw_flow.ndim != 1 or raw_flow.size != n_complete_weeks * HOURS_PER_WEEK:
        raise ValueError("raw_flow must contain exactly n_complete_weeks complete weeks")
    if not 0 <= shift_weeks < n_complete_weeks:
        raise ValueError("shift_weeks must lie in [0, n_complete_weeks)")
    signs = np.sign(raw_flow).reshape(n_complete_weeks, HOURS_PER_WEEK)
    magnitudes = np.abs(raw_flow).reshape(n_complete_weeks, HOURS_PER_WEEK)
    return (signs * np.roll(magnitudes, -shift_weeks, axis=0)).reshape(-1)


def _stratum_mean(values: np.ndarray, plan: _StratumPlan) -> np.ndarray:
    means = np.bincount(plan.codes, weights=values) / plan.counts
    return means[plan.codes]


def _windowed(values: np.ndarray, window: int) -> np.ndarray:
    n = values.size // window
    if n * window != values.size:
        raise ValueError("the series must contain an integer number of windows")
    return values.reshape(n, window)


def global_mbar_components(
    raw_flow: np.ndarray,
    strata: np.ndarray,
    window: int = HOURS_PER_WEEK,
) -> CrossComponents:
    """Cross decomposition using one global magnitude reference.

    All inputs are first recentered in the operational strata.  Thus this is
    the historical global-``mbar`` attribution diagnostic, not an alternative
    preprocessing rule.
    """
    plan = _stratum_plan(np.asarray(strata), window)
    return _global_mbar_components(raw_flow, plan)


def _global_mbar_components(raw_flow: np.ndarray, plan: _StratumPlan) -> CrossComponents:
    eps = _recenter(np.asarray(raw_flow, dtype=float), plan)
    signs, magnitudes = np.sign(eps), np.abs(eps)
    q = _windowed(eps, plan.window).sum(axis=1)
    m = _windowed(magnitudes, plan.window)
    s = _windowed(signs, plan.window)
    diagonal = (m**2).sum(axis=1)
    mbar = float(magnitudes.mean())
    sign = mbar**2 * (s.sum(axis=1) ** 2 - (s**2).sum(axis=1))
    cross = q**2 - diagonal - sign
    return CrossComponents(
        diagonal=float(diagonal.mean()),
        sign=float(sign.mean()),
        cross=float(cross.mean()),
        cross_uncorrected=float(cross.mean()),
        recentering_correction=0.0,
        q2=float((q**2).mean()),
    )


def null_aligned_components(
    raw_flow: np.ndarray,
    strata: np.ndarray,
    window: int = HOURS_PER_WEEK,
) -> CrossComponents:
    """Operational-null decomposition with a stratum-specific magnitude mean."""
    plan = _stratum_plan(np.asarray(strata), window)
    return _null_aligned_components(raw_flow, plan)


def _expected_recentered_magnitude_second_moment(
    flow: np.ndarray, plan: _StratumPlan
) -> np.ndarray:
    """Vectorised evaluation of the finite-population expectation in Eq. (8)."""
    signs, magnitudes = np.sign(flow), np.abs(flow)
    codes, counts = plan.codes, plan.counts.astype(float)
    sum_s = np.bincount(codes, weights=signs)
    sum_s2 = np.bincount(codes, weights=signs**2)
    sum_m = np.bincount(codes, weights=magnitudes)
    sum_m2 = np.bincount(codes, weights=magnitudes**2)
    sbar = sum_s / counts
    mbar = sum_m / counts
    sigma2 = np.maximum(0.0, sum_m2 / counts - mbar**2)
    sign_energy = sum_s2 - counts * sbar**2
    non_singleton = counts > 1
    variance_of_center = np.zeros_like(counts)
    variance_of_center[non_singleton] = (
        sigma2[non_singleton] * sign_energy[non_singleton]
        / (counts[non_singleton] * (counts[non_singleton] - 1))
    )
    covariance = np.zeros_like(flow)
    mask = non_singleton[codes]
    covariance[mask] = (
        signs[mask] * sigma2[codes[mask]]
        * (signs[mask] - sbar[codes[mask]])
        / (counts[codes[mask]] - 1)
    )
    mean_contribution = (signs - sbar[codes]) * mbar[codes]
    variance_contribution = (
        signs**2 * sigma2[codes] + variance_of_center[codes] - 2.0 * covariance
    )
    means = _windowed(mean_contribution, plan.window).sum(axis=1)
    variances = _windowed(variance_contribution, plan.window).sum(axis=1)
    return means**2 + variances


def _null_aligned_components(raw_flow: np.ndarray, plan: _StratumPlan) -> CrossComponents:
    eps = _recenter(np.asarray(raw_flow, dtype=float), plan)
    signs, magnitudes = np.sign(eps), np.abs(eps)
    q = _windowed(eps, plan.window).sum(axis=1)
    m = _windowed(magnitudes, plan.window)
    u = _windowed(signs * _stratum_mean(magnitudes, plan), plan.window)
    diagonal = (m**2).sum(axis=1)
    sign = u.sum(axis=1) ** 2 - (u**2).sum(axis=1)
    cross_uncorrected = q**2 - diagonal - sign
    q2_magnitude_null = _expected_recentered_magnitude_second_moment(eps, plan)
    correction = q2_magnitude_null - diagonal - sign
    cross = cross_uncorrected - correction
    return CrossComponents(
        diagonal=float(diagonal.mean()),
        sign=float(sign.mean()),
        cross=float(cross.mean()),
        cross_uncorrected=float(cross_uncorrected.mean()),
        recentering_correction=float(correction.mean()),
        q2=float((q**2).mean()),
    )


def _distribution_summary(values: np.ndarray, observed: float) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError("distribution and observed value must be finite and nonempty")
    median = float(np.median(values))
    return {
        "n_shifts": int(values.size),
        "minimum": float(values.min()),
        "q025": float(np.quantile(values, 0.025)),
        "q25": float(np.quantile(values, 0.25)),
        "median": median,
        "q75": float(np.quantile(values, 0.75)),
        "q975": float(np.quantile(values, 0.975)),
        "maximum": float(values.max()),
        "tail_fraction_abs_zero": float(
            (1 + np.count_nonzero(np.abs(values) >= abs(observed))) / (values.size + 1)
        ),
        "observed_minus_surrogate_median": float(observed - median),
        "tail_fraction_from_surrogate_median": float(
            (1 + np.count_nonzero(np.abs(values - median) >= abs(observed - median)))
            / (values.size + 1)
        ),
        "observed_over_surrogate_median": (
            float(observed / median) if median != 0 else None
        ),
        "surrogate_median_over_observed": (
            float(median / observed) if observed != 0 else None
        ),
    }


def exhaustive_week_shift_diagnostic(
    raw_flow: np.ndarray,
    strata: np.ndarray,
    n_complete_weeks: int,
) -> dict:
    """Evaluate each nonzero whole-week displacement once under both conventions."""
    raw_flow = np.asarray(raw_flow, dtype=float)
    if n_complete_weeks < 3:
        raise ValueError("at least three complete weeks are required")
    if raw_flow.size != n_complete_weeks * HOURS_PER_WEEK:
        raise ValueError("n_complete_weeks does not match raw_flow")
    if not np.all(np.isfinite(raw_flow)):
        raise ValueError("raw_flow must be finite")

    plan = _stratum_plan(np.asarray(strata), HOURS_PER_WEEK)
    observed_global = _global_mbar_components(raw_flow, plan)
    observed_aligned = _null_aligned_components(raw_flow, plan)
    rows = []
    for shift in range(1, n_complete_weeks):
        surrogate = misalign_magnitudes_by_weeks(raw_flow, n_complete_weeks, shift)
        global_ = _global_mbar_components(surrogate, plan)
        aligned = _null_aligned_components(surrogate, plan)
        rows.append({
            "shift_weeks": shift,
            "cross_global_mbar": global_.cross,
            "cross_null_aligned": aligned.cross,
        })

    global_values = np.array([row["cross_global_mbar"] for row in rows])
    aligned_values = np.array([row["cross_null_aligned"] for row in rows])
    return {
        "definition": (
            "Each nonzero integer-week circular shift uses eps'_t=s_t m_{t+Delta}; "
            "each surrogate is recentered within the original quarter-by-hour-of-week strata."
        ),
        "n_complete_weeks": int(n_complete_weeks),
        "legal_nonzero_shifts": int(n_complete_weeks - 1),
        "observed": {
            "global_mbar": asdict(observed_global),
            "null_aligned": asdict(observed_aligned),
        },
        "summaries": {
            "global_mbar_cross": _distribution_summary(global_values, observed_global.cross),
            "null_aligned_cross": _distribution_summary(aligned_values, observed_aligned.cross),
        },
        "shifts": rows,
    }


def synthetic_control_raw(
    n_complete_weeks: int,
    seed: int,
    coupled: bool,
    coupling_strength: float = 0.9,
    coupling_mode: str = "run",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a control with persistent sign and magnitude paths.

    In the coupled control, magnitude depends either on a short sign-run score
    or on the absolute weekly sign imbalance.  The latter is the positive
    control matched to the week-shift diagnostic's resolution.
    Both controls have the same Markov sign persistence, magnitude AR(1)
    persistence, and hour-of-week modulation.  The controls calibrate whether
    the rank diagnostic can distinguish a declared sign--size synchrony from
    independent marginal memories; they are not a generative market model.
    """
    if n_complete_weeks < 16:
        raise ValueError("synthetic controls require at least sixteen weeks")
    rng = np.random.default_rng(seed)
    n = n_complete_weeks * HOURS_PER_WEEK
    signs = np.empty(n, dtype=float)
    signs[0] = rng.choice((-1.0, 1.0))
    flips = rng.random(n - 1) >= 0.91
    for i, flip in enumerate(flips, start=1):
        signs[i] = -signs[i - 1] if flip else signs[i - 1]
    latent = np.empty(n, dtype=float)
    latent[0] = rng.normal()
    noise = rng.normal(size=n - 1)
    for i, innovation in enumerate(noise, start=1):
        latent[i] = 0.94 * latent[i - 1] + innovation
    hour = np.arange(n) % HOURS_PER_WEEK
    seasonal = 0.25 * np.cos(2 * np.pi * hour / HOURS_PER_WEEK)
    run = np.zeros(n, dtype=float)
    for lag, weight in ((1, 0.55), (2, 0.30), (3, 0.15)):
        run[lag:] += weight * signs[lag:] * signs[:-lag]
    if coupling_strength < 0:
        raise ValueError("coupling_strength must be nonnegative")
    if coupling_mode not in {"run", "weekly_imbalance", "weekly_amplifier"}:
        raise ValueError(
            "coupling_mode must be 'run', 'weekly_imbalance', or 'weekly_amplifier'"
        )
    if not coupled:
        interaction = 0.0
    elif coupling_mode == "run":
        interaction = coupling_strength * run
    elif coupling_mode == "weekly_imbalance":
        weekly = np.abs(signs.reshape(n_complete_weeks, HOURS_PER_WEEK).mean(axis=1))
        weekly = (weekly - weekly.mean()) / weekly.std(ddof=0)
        interaction = coupling_strength * np.repeat(weekly, HOURS_PER_WEEK)
    else:
        weekly = np.abs(signs.reshape(n_complete_weeks, HOURS_PER_WEEK).mean(axis=1))
        interaction = np.log1p(coupling_strength * np.repeat(weekly, HOURS_PER_WEEK))
    magnitudes = np.exp(0.35 * latent + seasonal + interaction)
    raw = signs * magnitudes
    quarter = np.arange(n) // (13 * HOURS_PER_WEEK)
    strata = quarter * HOURS_PER_WEEK + hour
    if coupled and coupling_mode == "weekly_amplifier":
        raw = recenter_within_strata(raw, strata)
    return raw, strata
