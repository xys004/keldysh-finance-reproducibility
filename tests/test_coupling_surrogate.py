from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.coupling_surrogate import (
    _expected_recentered_magnitude_second_moment,
    _stratum_plan,
    exhaustive_week_shift_diagnostic,
    misalign_magnitudes_by_weeks,
    recenter_within_strata,
    synthetic_control_raw,
)
from keldysh_finance.fano_validation import (
    HOURS_PER_WEEK,
    expected_recentered_magnitude_second_moment,
)


def _toy_raw(weeks: int = 20) -> tuple[np.ndarray, np.ndarray]:
    raw, strata = synthetic_control_raw(weeks, seed=17, coupled=False)
    return raw, strata


def test_zero_shift_is_identity_and_nonzero_shift_preserves_both_paths():
    raw, _ = _toy_raw()
    weeks = raw.size // HOURS_PER_WEEK
    zero = misalign_magnitudes_by_weeks(raw, weeks, 0)
    shifted = misalign_magnitudes_by_weeks(raw, weeks, 3)
    assert np.array_equal(zero, raw)
    assert np.array_equal(np.sort(np.abs(shifted)), np.sort(np.abs(raw)))
    assert np.array_equal(np.sign(shifted), np.sign(raw))


def test_recentering_is_stratum_exact_and_detects_a_broken_convention():
    raw, strata = _toy_raw()
    centered = recenter_within_strata(raw, strata)
    _, codes = np.unique(strata, return_inverse=True)
    means = np.bincount(codes, weights=centered) / np.bincount(codes)
    assert np.max(np.abs(means)) < 1e-13
    # Mutation guard: omitting the recentering leaves an intentionally visible bias.
    raw_means = np.bincount(codes, weights=raw) / np.bincount(codes)
    assert np.max(np.abs(raw_means)) > 1e-4


def test_exhaustive_distribution_has_every_nonzero_shift_once():
    raw, strata = _toy_raw()
    weeks = raw.size // HOURS_PER_WEEK
    result = exhaustive_week_shift_diagnostic(raw, strata, weeks)
    shifts = [row["shift_weeks"] for row in result["shifts"]]
    assert result["legal_nonzero_shifts"] == weeks - 1
    assert shifts == list(range(1, weeks))
    assert result["summaries"]["global_mbar_cross"]["n_shifts"] == weeks - 1
    assert result["summaries"]["null_aligned_cross"]["n_shifts"] == weeks - 1


def test_vectorised_recentring_expectation_matches_independent_formula():
    raw, strata = _toy_raw()
    centered = recenter_within_strata(raw, strata)
    plan = _stratum_plan(strata, HOURS_PER_WEEK)
    fast = _expected_recentered_magnitude_second_moment(centered, plan)
    reference = expected_recentered_magnitude_second_moment(
        centered, strata, window=HOURS_PER_WEEK
    )
    assert np.allclose(fast, reference, rtol=0, atol=1e-11)


def test_injected_synchrony_changes_the_cross_diagnostic():
    independent_raw, strata = synthetic_control_raw(40, seed=31, coupled=False)
    coupled_raw, _ = synthetic_control_raw(40, seed=31, coupled=True)
    independent = exhaustive_week_shift_diagnostic(independent_raw, strata, 40)
    coupled = exhaustive_week_shift_diagnostic(coupled_raw, strata, 40)
    independent_tail = independent["summaries"]["null_aligned_cross"]["tail_fraction_from_surrogate_median"]
    coupled_tail = coupled["summaries"]["null_aligned_cross"]["tail_fraction_from_surrogate_median"]
    assert coupled_tail < independent_tail
