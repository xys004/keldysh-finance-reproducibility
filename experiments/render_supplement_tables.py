"""Render Supplemental Material tables from deposited JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
TARGET = ROOT / "paper" / "generated_referee_tables.tex"


def _load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _pvalue(value: float) -> str:
    return "$" + f"{value:.4f}" + "$" if value >= 0.0001 else r"$<10^{-4}$"


def _close(lines: list[str], wide: bool = False) -> None:
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}",
                  r"\end{table*}" if wide else r"\end{table}", ""])


def main() -> None:
    exp11 = _load("exp11_haar_crossover.json")
    exp14 = _load("exp14_referee_robustness.json")
    exp15 = _load("exp15_flow_null_decomposition.json")
    exp16 = _load("exp16_scaling_reassessment.json")
    lines: list[str] = []

    lines.extend([
        r"\begin{table}[t]", r"\caption{Standardised cumulants at the endpoints of the dyadic scale grid. Intervals in Fig.~\ref{fig:supp-cumulants} use the full grid.}",
        r"\label{tab:supp-cumulants}", r"\begin{ruledtabular}", r"\begin{tabular}{lrrrr}",
        r" & $\gamma_1(1)$ & $\gamma_1(128)$ & $\gamma_2(1)$ & $\gamma_2(128)$ \\",
        r"\colrule",
    ])
    for row in exp14["results"]:
        first = row["standardized_cumulants"][0]
        last = row["standardized_cumulants"][-1]
        lines.append(
            f"{row['asset']} & {first['gamma1']:.2f} & {last['gamma1']:.2f} & "
            f"{first['gamma2']:.2f} & {last['gamma2']:.2f} " + r"\\"
        )
    _close(lines)

    lines.extend([
        r"\begin{table*}[t]", r"\caption{Decomposition of weekly variance amplification under quarterly $\times$ hour-of-week conditioning. $R_J$, $R_S$, and $R_M$ divide observed variance by the median joint-order, sign-order, and magnitude-order null. The interval is an eight-week moving-block bootstrap interval for $R_J$, conditional on its null median.}",
        r"\label{tab:supp-decomposition}", r"\begin{ruledtabular}", r"\begin{tabular}{llrrrr}",
        r"asset & flow & $R_J$ & $R_S$ & $R_M$ & $95\%$ interval for $R_J$ \\",
        r"\colrule",
    ])
    for row_index, row in enumerate(exp15["results"]):
        if row_index == 4:
            lines.append(r"\colrule")
        for mode in row["flow_definitions"]:
            estimates = {v["null_mode"]: v for v in mode["primary_quarterly"]}
            ci = estimates["joint"]["weekly_block_bootstrap"][1]
            lines.append(
                f"{row['asset']} & {mode['flow_mode']} & "
                f"{estimates['joint']['variance_ratio_to_null_median']:.2f} & "
                f"{estimates['sign']['variance_ratio_to_null_median']:.2f} & "
                f"{estimates['magnitude']['variance_ratio_to_null_median']:.2f} & "
                f"[{ci['ratio_ci025']:.2f},{ci['ratio_ci975']:.2f}] " + r"\\"
            )
    _close(lines, wide=True)

    lines.extend([
        r"\begin{table*}[t]", r"\footnotesize", r"\caption{Sensitivity of the joint weekly variance null. $R_{50}=V_{\rm obs}/\mathrm{median}(V_{\rm null})$ and $R_{98.75}=V_{\rm obs}/Q_{0.9875}(V_{\rm null})$. Primary quarterly rows use 9999 permutations; monthly and biweekly rows use 1999.}",
        r"\label{tab:supp-null}", r"\begin{ruledtabular}", r"\begin{tabular}{lllrrrr}",
        r"asset & flow & conditioning & $R_{50}$ & $R_{98.75}$ & $p_{\rm emp}$ & fixed fraction \\",
        r"\colrule",
    ])
    primary_assets = set(exp15["method"]["asset_panel"]["original"])
    for row in exp15["results"]:
        if row["asset"] not in primary_assets:
            continue
        for mode in row["flow_definitions"]:
            nulls = [next(v for v in mode["primary_quarterly"]
                          if v["null_mode"] == "joint")]
            nulls.extend(mode["joint_stratification_sensitivity"])
            for null in nulls:
                lines.append(
                    f"{row['asset']} & {mode['flow_mode']} & {null['stratification']} & "
                    f"{null['variance_ratio_to_null_median']:.2f} & "
                    f"{null['variance_ratio_to_null_quantile']:.2f} & "
                    f"{_pvalue(null['empirical_p_upper'])} & "
                    f"{100 * null['singleton_fraction']:.2f}\\% " + r"\\"
                )
    _close(lines, wide=True)

    wavelets = {row["activo"]: row for row in exp11["reales"]}
    lines.extend([
        r"\begin{table}[t]", r"\caption{Global exponent $\nu$ over $T=8,\ldots,256$. db1 removes a constant, db2 a linear trend, and db3 a quadratic trend.}",
        r"\label{tab:supp-wavelets}", r"\begin{ruledtabular}", r"\begin{tabular}{lrrrr}",
        r"asset & block variance & db1 & db2 & db3 \\",
        r"\colrule",
    ])
    for asset in ("BTC", "ETH", "BNB", "SOL"):
        row = wavelets[asset]
        lines.append(
            f"{asset} & {row['bloques']['nu_global']:.2f} & "
            f"{row['db1']['nu_global']:.2f} & {row['db2']['nu_global']:.2f} & "
            f"{row['db3']['nu_global']:.2f} " + r"\\"
        )
    _close(lines)

    lines.extend([
        r"\FloatBarrier", r"\refstepcounter{table}", r"\label{tab:supp-scaling}",
        r"\noindent\textsc{Table \thetable.} Asset-specific scaling reassessment. The consistency band maps $C(\tau)\sim\tau^{-a}$ to $\nu=2-a$ only for $a<1$; $a>1$ maps to diffusion. Amplitudes are not fitted, so the band is not a quantitative prediction.",
        r"\begin{center}", r"\begin{ruledtabular}", r"\begin{tabular}{lrrrrr}",
        r"asset & $a_f$ & $a_s$ & consistency band & $\nu_{1h}$ & $\nu_{4h}$ \\",
        r"\colrule",
    ])
    for row in exp16["two_regime_asset_checks"]:
        block = {v["series"].split("|")[1]: v["nu"] for v in row["block_exponents"]}
        band = row["finite_scale_consistency_band"]
        lines.append(
            f"{row['asset']} & {row['acf_fast']['a']:.2f} & "
            f"{row['acf_slow']['a']:.2f} & [{band[0]:.2f},{band[1]:.2f}] & "
            f"{block['1h']:.2f} & {block['4h']:.2f} " + r"\\"
        )
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{center}", ""])

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
