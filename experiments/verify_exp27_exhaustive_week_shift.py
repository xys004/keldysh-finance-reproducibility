"""Independent audit of the EXP 27 week-shift artifact.

This verifier deliberately does not call ``coupling_surrogate`` for the market
calculation.  It reconstructs selected rows from the raw CSVs with pandas
grouping and the established finite-population expectation, then proves that a
reversed shift convention changes the checked result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keldysh_finance.fano_validation import (  # noqa: E402
    HOURS_PER_WEEK,
    _stratum_period,
    expected_recentered_magnitude_second_moment,
    prepare_complete_weeks,
)


CHECKED_SHIFTS = (1, 37, 103, 205)
TOLERANCE = 2e-10


def load(symbol: str) -> pd.DataFrame:
    path = ROOT / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    frame = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.dropna()


def labels(prepared) -> np.ndarray:
    period = _stratum_period(prepared, "quarterly")
    return period.astype(np.int64) * HOURS_PER_WEEK + prepared.hour_of_week.astype(np.int64)


def recenter(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    return values - pd.Series(values).groupby(strata, sort=False).transform("mean").to_numpy()


def shifted_raw(values: np.ndarray, weeks: int, shift: int, reverse: bool = False) -> np.ndarray:
    sign = np.sign(values).reshape(weeks, HOURS_PER_WEEK)
    magnitude = np.abs(values).reshape(weeks, HOURS_PER_WEEK)
    direction = shift if reverse else -shift
    return (sign * np.roll(magnitude, direction, axis=0)).reshape(-1)


def components(raw: np.ndarray, strata: np.ndarray) -> tuple[float, float]:
    eps = recenter(raw, strata)
    signs, magnitudes = np.sign(eps), np.abs(eps)
    e = eps.reshape(-1, HOURS_PER_WEEK)
    s = signs.reshape(-1, HOURS_PER_WEEK)
    m = magnitudes.reshape(-1, HOURS_PER_WEEK)
    q2 = (e.sum(axis=1) ** 2)
    diagonal = (m**2).sum(axis=1)
    global_sign = magnitudes.mean() ** 2 * (s.sum(axis=1) ** 2 - (s**2).sum(axis=1))
    global_cross = float((q2 - diagonal - global_sign).mean())
    mbar_h = pd.Series(magnitudes).groupby(strata, sort=False).transform("mean").to_numpy()
    u = (signs * mbar_h).reshape(-1, HOURS_PER_WEEK)
    sign_aligned = u.sum(axis=1) ** 2 - (u**2).sum(axis=1)
    uncorrected = q2 - diagonal - sign_aligned
    correction = (
        expected_recentered_magnitude_second_moment(eps, strata, window=HOURS_PER_WEEK)
        - diagonal - sign_aligned
    )
    return global_cross, float((uncorrected - correction).mean())


def close(actual: float, expected: float) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return abs(actual - expected) <= TOLERANCE * scale


def main() -> None:
    artifact_path = ROOT / "output" / "exp27_exhaustive_week_shift.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    failures = []
    mutation_detected = False
    for row in artifact["market"]:
        prepared = prepare_complete_weeks(load(f"{row['asset']}USDT"), flow_mode=row["flow_mode"])
        strata = labels(prepared)
        weeks = prepared.n_complete_weeks
        if row["legal_nonzero_shifts"] != weeks - 1:
            failures.append(f"{row['asset']}|{row['flow_mode']}: legal shift count mismatch")
            continue
        by_shift = {item["shift_weeks"]: item for item in row["shifts"]}
        for shift in CHECKED_SHIFTS:
            if shift >= weeks:
                continue
            global_, aligned = components(shifted_raw(prepared.flow, weeks, shift), strata)
            recorded = by_shift[shift]
            if not close(global_, recorded["cross_global_mbar"]):
                failures.append(f"{row['asset']}|{row['flow_mode']}|{shift}: global mismatch")
            if not close(aligned, recorded["cross_null_aligned"]):
                failures.append(f"{row['asset']}|{row['flow_mode']}|{shift}: aligned mismatch")
            if shift == 1:
                reversed_global, _ = components(
                    shifted_raw(prepared.flow, weeks, shift, reverse=True), strata
                )
                if not close(reversed_global, recorded["cross_global_mbar"]):
                    mutation_detected = True

    if not mutation_detected:
        failures.append("shift-direction mutation was not detected")
    expected_gate = all(
        c["passes_global"] and c["passes_aligned"]
        for c in (artifact["controls"][0], artifact["controls"][2])
    )
    if artifact["control_gate"] != expected_gate:
        failures.append("control gate does not match its declared constituent controls")
    result = {
        "artifact": str(artifact_path),
        "checked_shifts": list(CHECKED_SHIFTS),
        "tolerance_relative": TOLERANCE,
        "mutation_detected": mutation_detected,
        "control_gate": artifact["control_gate"],
        "failures": failures,
        "passes": not failures,
    }
    path = ROOT / "output" / "verify_exp27_exhaustive_week_shift.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("EXP 27 independent verification failed")


if __name__ == "__main__":
    main()
