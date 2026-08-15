from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keldysh_finance.fano_validation import (paired_normalization_share_difference,
                                             prepare_complete_weeks,
                                             stratified_fano_memory_test,
                                             stratified_sign_size_decomposition,
                                             stratified_variance_memory_test,
                                             validate_hourly_klines)


def _frame(weeks: int = 16, seed: int = 1, persistent: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = weeks * 168
    idx = pd.date_range("2025-01-06", periods=n, freq="h", tz="UTC")
    trades = rng.integers(20, 80, size=n).astype(float)
    volume = rng.lognormal(2.0, 0.8, size=n)
    if persistent:
        sign = np.empty(n)
        sign[0] = 1.0
        for k in range(1, n):
            sign[k] = sign[k - 1] if rng.random() < 0.985 else -sign[k - 1]
        imbalance = 0.85 * sign + 0.05 * rng.normal(size=n)
    else:
        imbalance = rng.uniform(-0.9, 0.9, size=n)
    imbalance = np.clip(imbalance, -0.99, 0.99)
    buy = 0.5 * volume * (1.0 + imbalance)
    return pd.DataFrame({"Volume": volume, "tbBase": buy, "trades": trades}, index=idx)


def test_validation_rejects_duplicate_hours():
    df = _frame()
    bad = pd.concat([df, df.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        validate_hourly_klines(bad)


def test_validation_rejects_incompatible_buy_volume():
    df = _frame()
    df.iloc[0, df.columns.get_loc("tbBase")] = 2.0 * df.iloc[0]["Volume"]
    with pytest.raises(ValueError, match="tbBase"):
        validate_hourly_klines(df)


def test_stratified_null_is_near_one_without_memory():
    prepared = prepare_complete_weeks(_frame(weeks=32, seed=2, persistent=False))
    result = stratified_fano_memory_test(prepared, permutations=399, seed=3, batch_size=64)
    assert 0.65 < result["variance_ratio_to_null_median"] < 1.5
    assert not result["memory_majority_gate"]


def test_stratified_null_detects_strong_memory():
    prepared = prepare_complete_weeks(_frame(weeks=32, seed=4, persistent=True))
    result = stratified_fano_memory_test(prepared, permutations=399, seed=5, batch_size=64)
    assert result["variance_ratio_to_null_median"] > 2.0


@pytest.mark.parametrize("stratification", ["quarterly", "monthly", "biweekly"])
def test_all_stratifications_preserve_a_memory_free_null(stratification):
    prepared = prepare_complete_weeks(_frame(weeks=32, seed=6, persistent=False))
    result = stratified_fano_memory_test(
        prepared,
        permutations=199,
        seed=7,
        batch_size=64,
        stratification=stratification,
    )
    assert result["stratification"] == stratification
    assert 0.55 < result["variance_ratio_to_null_median"] < 1.65
    assert 0.0 <= result["singleton_fraction"] < 0.1
    assert result["min_stratum_size"] >= 1
    assert result["max_stratum_size"] >= 2


def test_unknown_stratification_is_rejected():
    prepared = prepare_complete_weeks(_frame(weeks=16, seed=8))
    with pytest.raises(ValueError, match="stratification"):
        stratified_fano_memory_test(
            prepared, permutations=99, stratification="daily"
        )


def test_normalized_flow_matches_declared_imbalance():
    df = _frame(weeks=16, seed=9)
    prepared = prepare_complete_weeks(df, flow_mode="normalized")
    expected = (2.0 * df.loc[prepared.index, "tbBase"].to_numpy()
                / df.loc[prepared.index, "Volume"].to_numpy() - 1.0)
    assert prepared.flow_mode == "normalized"
    assert np.allclose(prepared.flow, expected)
    with pytest.raises(ValueError, match="raw signed volume"):
        stratified_fano_memory_test(prepared, permutations=99)


@pytest.mark.parametrize("null_mode", ["joint", "sign", "magnitude"])
def test_decomposed_nulls_are_finite_and_bootstrapped(null_mode):
    prepared = prepare_complete_weeks(
        _frame(weeks=32, seed=10, persistent=True), flow_mode="normalized"
    )
    result = stratified_variance_memory_test(
        prepared,
        permutations=199,
        seed=11,
        batch_size=32,
        null_mode=null_mode,
        bootstrap_replicates=199,
        bootstrap_block_lengths=(4, 8),
    )
    assert result["flow_mode"] == "normalized"
    assert result["null_mode"] == null_mode
    assert np.isfinite(result["variance_ratio_to_null_median"])
    assert len(result["weekly_block_bootstrap"]) == 2
    assert "fano_observed" not in result


def test_exact_sign_size_decomposition_reconstructs_weekly_variance():
    prepared = prepare_complete_weeks(
        _frame(weeks=48, seed=12, persistent=True), flow_mode="normalized"
    )
    result = stratified_sign_size_decomposition(
        prepared,
        bootstrap_replicates=199,
        bootstrap_block_lengths=(4, 8),
        seed=13,
    )
    assert result["exact_reconstruction_error"] < 1e-10
    assert result["max_weekly_reconstruction_relative_error"] < 1e-12
    assert abs(
        result["sign_share_of_excess"]
        + result["coupling_share_of_excess"] - 1.0
    ) < 1e-12
    assert abs(result["weekly_flow_mean"]) < 1e-12
    assert abs(result["max_stratum_flow_mean"]) < 1e-12
    assert abs(result["max_stratum_sign_proxy_mean"]) < 1e-12
    assert len(result["paired_weekly_block_bootstrap"]) == 2


def test_joint_null_mean_matches_analytic_marginal_reference():
    prepared = prepare_complete_weeks(
        _frame(weeks=48, seed=14, persistent=True), flow_mode="normalized"
    )
    decomposition = stratified_sign_size_decomposition(prepared)
    null = stratified_variance_memory_test(
        prepared,
        permutations=1999,
        seed=15,
        batch_size=64,
        null_mode="joint",
    )
    assert np.isclose(
        null["variance_null_analytic_expectation"],
        decomposition["marginal_reference"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert abs(null["null_mean_relative_error_to_analytic"]) < 0.05


def test_normalization_share_difference_uses_paired_blocks():
    frame = _frame(weeks=48, seed=16, persistent=True)
    raw = prepare_complete_weeks(frame, flow_mode="raw")
    normalized = prepare_complete_weeks(frame, flow_mode="normalized")
    result = paired_normalization_share_difference(
        raw,
        normalized,
        bootstrap_replicates=199,
        bootstrap_block_lengths=(4, 8),
        seed=17,
    )
    assert np.isfinite(result["delta_sign_share"])
    assert len(result["paired_weekly_block_bootstrap"]) == 2
    assert all(
        row["valid_paired_replicates"] >= 100
        for row in result["paired_weekly_block_bootstrap"]
    )
