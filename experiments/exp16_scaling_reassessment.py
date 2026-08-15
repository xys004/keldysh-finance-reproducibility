"""Reassess the finite-scale variance exponents asset by asset.

This script is intentionally a reassessment of already archived measurements.  It
does not treat the ACF and block-sum variance as independent evidence: the
finite-sample identity links them exactly.  It reports (i) failure of a single
power-law ACF prediction for every asset/interval and (ii) the weaker
consistency band implied by two fitted ACF regimes with the correct a>=1
domain, where summable correlations give diffusion (with a logarithmic
boundary at a=1).
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exp16_scaling_reassessment.json"


def _load(name: str) -> dict:
    return json.loads((ROOT / "output" / name).read_text(encoding="utf-8"))


def _nu_from_a(a: float) -> dict:
    if a < 1.0:
        return {"regime": "non-summable", "nu": 2.0 - a}
    if a > 1.0:
        return {"regime": "summable", "nu": 1.0}
    return {"regime": "marginal", "nu": 1.0, "log_correction": "T log T"}


def main() -> None:
    d6 = _load("exp06_conteo_flujo.json")
    d9 = _load("exp09_msrjd_orden2.json")
    d11 = _load("exp11_haar_crossover.json")

    series_rows = []
    for row in d6["resultados"]:
        prediction = float(row["prediccion_pendiente_k2"])
        observed = float(row["pendiente_k2_full"]["pendiente"])
        se = float(row["pendiente_k2_full"]["se"])
        series_rows.append({
            "series": row["clave"],
            "single_acf_prediction": prediction,
            "observed_block_variance_exponent": observed,
            "observed_standard_error": se,
            "difference": observed - prediction,
            "z_using_observed_se_only": (observed - prediction) / se,
            "compatible_with_single_exponent_at_3se": abs(observed - prediction) <= 3 * se,
        })

    wavelet = {row["activo"]: row for row in d11["reales"]}
    asset_rows = []
    for asset, regimes in d9["dos_regimenes"].items():
        raw = regimes["crudo"]
        fast = raw["[2,10]"]
        slow = raw["[10,50]"]
        a_fast = -float(fast["pendiente"])
        a_slow = -float(slow["pendiente"])
        mapped_fast = _nu_from_a(a_fast)
        mapped_slow = _nu_from_a(a_slow)
        band = sorted([float(mapped_fast["nu"]), float(mapped_slow["nu"])])
        measured = [r for r in series_rows if r["series"].startswith(asset + "|")]
        w = wavelet[asset]
        asset_rows.append({
            "asset": asset,
            "acf_fast": {"a": a_fast, "se": float(fast["se"]), **mapped_fast},
            "acf_slow": {"a": a_slow, "se": float(slow["se"]), **mapped_slow},
            "finite_scale_consistency_band": band,
            "block_exponents": [
                {
                    "series": r["series"],
                    "nu": r["observed_block_variance_exponent"],
                    "se": r["observed_standard_error"],
                    "inside_consistency_band": (
                        band[0] <= r["observed_block_variance_exponent"] <= band[1]
                    ),
                }
                for r in measured
            ],
            "drift_blind_global_exponents": {
                key: float(w[key]["nu_global"]) for key in ("db1", "db2", "db3")
            },
            "all_block_exponents_inside_band": bool(all(
                band[0] <= r["observed_block_variance_exponent"] <= band[1]
                for r in measured
            )),
        })

    payload = {
        "convention": "C_epsilon(tau) ~ tau^(-a)",
        "asymptotic_map": {
            "0<a<1": "Var(Q_T) ~ T^(2-a)",
            "a=1": "Var(Q_T) ~ T log T",
            "a>1": "Var(Q_T) ~ T",
        },
        "logical_status": {
            "acf_variance_relation": (
                "exact second-order identity, not independent corroboration"
            ),
            "single_power_model": (
                "rejected descriptively for all eight series on the archived "
                "fit ranges; z values omit uncertainty in the ACF prediction"
            ),
            "two_regime_band": (
                "finite-scale consistency check only; it is not a fitted "
                "prediction because regime amplitudes and crossover are absent"
            ),
        },
        "single_power_series_tests": series_rows,
        "two_regime_asset_checks": asset_rows,
        "all_single_power_tests_pass": bool(all(
            row["compatible_with_single_exponent_at_3se"] for row in series_rows
        )),
        "all_assets_inside_two_regime_band": bool(all(
            row["all_block_exponents_inside_band"] for row in asset_rows
        )),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    for row in series_rows:
        print(
            f"{row['series']:<7} pred={row['single_acf_prediction']:.3f} "
            f"obs={row['observed_block_variance_exponent']:.3f} "
            f"z_obs_only={row['z_using_observed_se_only']:.1f}"
        )
    for row in asset_rows:
        print(
            f"{row['asset']}: band={row['finite_scale_consistency_band']} "
            f"all_inside={row['all_block_exponents_inside_band']}"
        )


if __name__ == "__main__":
    main()
