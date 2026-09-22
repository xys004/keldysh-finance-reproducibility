"""Exhaustive calibration of the week-shift sign--magnitude surrogate.

Acceptance criteria declared before reading the market output:

* all ``n_weeks-1`` nonzero week shifts are evaluated for every primary
  asset/observable and both cross conventions;
* an independent persistent synthetic control must be non-extreme
  (two-sided rank around the surrogate median > 0.05) in at least 9/12 seeds;
* a deliberately strong operational control with magnitude synchronized to the
  absolute weekly sign imbalance (lambda=8) must be extreme (tail fraction <=
  0.05) in
  at least 9/12 seeds.

The control verifies discrimination of this *diagnostic*.  It cannot identify
the microscopic origin of any market residual.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keldysh_finance.coupling_surrogate import (  # noqa: E402
    exhaustive_week_shift_diagnostic,
    synthetic_control_raw,
)
from keldysh_finance.fano_validation import _stratum_period, prepare_complete_weeks  # noqa: E402


ASSETS = ("BTC", "ETH", "BNB", "SOL")
FLOW_MODES = ("raw", "normalized")
N_CONTROL_SEEDS = 12
# Match the market experiment: 206 complete weeks and therefore 205 legal shifts.
CONTROL_WEEKS = 206
MIN_CONTROL_FRACTION = 0.75
WEAK_COUPLING_STRENGTH = 0.9
STRONG_COUPLING_STRENGTH = 8.0


def load(symbol: str) -> pd.DataFrame:
    path = ROOT / "output" / "cache" / f"flow_{symbol}_1h_4.0y.csv"
    frame = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame.dropna()


def quarterly_strata(prepared) -> np.ndarray:
    period = _stratum_period(prepared, "quarterly")
    return period.astype(np.int64) * 168 + prepared.hour_of_week.astype(np.int64)


def market_result(asset: str, flow_mode: str) -> dict:
    prepared = prepare_complete_weeks(load(f"{asset}USDT"), flow_mode=flow_mode)
    result = exhaustive_week_shift_diagnostic(
        prepared.flow, quarterly_strata(prepared), prepared.n_complete_weeks
    )
    result.update({"asset": asset, "flow_mode": flow_mode})
    return result


def control_result(
    coupled: bool,
    coupling_strength: float = 0.0,
    coupling_mode: str = "run",
) -> dict:
    rows = []
    for seed in range(N_CONTROL_SEEDS):
        raw, strata = synthetic_control_raw(
            CONTROL_WEEKS,
            seed=seed,
            coupled=coupled,
            coupling_strength=coupling_strength,
            coupling_mode=coupling_mode,
        )
        result = exhaustive_week_shift_diagnostic(raw, strata, CONTROL_WEEKS)
        rows.append({
            "seed": seed,
            "global_tail_fraction_from_median": result["summaries"]["global_mbar_cross"]["tail_fraction_from_surrogate_median"],
            "aligned_tail_fraction_from_median": result["summaries"]["null_aligned_cross"]["tail_fraction_from_surrogate_median"],
        })
    global_tail = np.array([row["global_tail_fraction_from_median"] for row in rows])
    aligned_tail = np.array([row["aligned_tail_fraction_from_median"] for row in rows])
    if coupled:
        global_rate = float(np.mean(global_tail <= 0.05))
        aligned_rate = float(np.mean(aligned_tail <= 0.05))
        criterion = "extreme"
    else:
        global_rate = float(np.mean(global_tail > 0.05))
        aligned_rate = float(np.mean(aligned_tail > 0.05))
        criterion = "non_extreme"
    return {
        "kind": (
            f"injected_{coupling_mode}_synchrony_lambda_{coupling_strength:g}"
            if coupled else "independent_persistent_paths"
        ),
        "coupling_strength": coupling_strength if coupled else None,
        "coupling_mode": coupling_mode if coupled else None,
        "criterion": criterion,
        "n_seeds": N_CONTROL_SEEDS,
        "global_rate": global_rate,
        "aligned_rate": aligned_rate,
        "required_rate": MIN_CONTROL_FRACTION,
        "passes_global": bool(global_rate >= MIN_CONTROL_FRACTION),
        "passes_aligned": bool(aligned_rate >= MIN_CONTROL_FRACTION),
        "rows": rows,
    }


def main() -> None:
    print("=" * 88)
    print(" EXP 27 - exhaustive week-shift cross diagnostic")
    print("=" * 88)
    market = []
    for flow_mode in FLOW_MODES:
        for asset in ASSETS:
            result = market_result(asset, flow_mode)
            market.append(result)
            g = result["summaries"]["global_mbar_cross"]
            a = result["summaries"]["null_aligned_cross"]
            print(
                f" {asset}|{flow_mode}: {result['legal_nonzero_shifts']} shifts; "
                f"global retention={g['surrogate_median_over_observed']:.3f}, "
                f"aligned retention={a['surrogate_median_over_observed']:.3f}"
            )

    controls = [
        control_result(coupled=False),
        control_result(coupled=True, coupling_strength=WEAK_COUPLING_STRENGTH),
        control_result(
            coupled=True,
            coupling_strength=STRONG_COUPLING_STRENGTH,
            coupling_mode="weekly_amplifier",
        ),
    ]
    for control in controls:
        print(
            f" control {control['kind']}: global={control['global_rate']:.2f}, "
            f"aligned={control['aligned_rate']:.2f}, required={control['required_rate']:.2f}"
        )
    output = {
        "pre_registered_acceptance": {
            "all_nonzero_week_shifts": True,
            "control_seeds": N_CONTROL_SEEDS,
            "control_weeks": CONTROL_WEEKS,
            "minimum_rate": MIN_CONTROL_FRACTION,
            "independent": "tail fraction around the surrogate median > 0.05",
            "strong_coupled": (
                "tail fraction around the surrogate median <= 0.05 with operationally centred magnitude tied to absolute weekly sign imbalance "
                f"and declared lambda={STRONG_COUPLING_STRENGTH:g}"
            ),
            "weak_coupled": (
                f"sensitivity diagnostic only, lambda={WEAK_COUPLING_STRENGTH:g}"
            ),
        },
        "market": market,
        "controls": controls,
        "control_gate": bool(
            controls[0]["passes_global"] and controls[0]["passes_aligned"]
            and controls[2]["passes_global"] and controls[2]["passes_aligned"]
        ),
    }
    path = ROOT / "output" / "exp27_exhaustive_week_shift.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f" log: {path}")
    if not output["control_gate"]:
        raise SystemExit("control calibration failed; do not interpret market ranks")


if __name__ == "__main__":
    main()
