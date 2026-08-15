"""Render LaTeX tables for paper3_supplement.tex from deposited JSON logs."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP14 = ROOT / "output" / "exp14_referee_robustness.json"
EXP11 = ROOT / "output" / "exp11_haar_crossover.json"
TARGET = ROOT / "paper" / "generated_referee_tables.tex"


def _pvalue(value: float) -> str:
    return rf"${value:.4f}$" if value >= 0.0001 else r"$<10^{-4}$"


def main() -> None:
    exp14 = json.loads(EXP14.read_text(encoding="utf-8"))
    exp11 = json.loads(EXP11.read_text(encoding="utf-8"))
    lines: list[str] = []

    lines.extend([
        r"\begin{table}[t]",
        r"\caption{Standardised cumulants at the endpoints of the dyadic scale grid. Intervals in Fig.~\ref{fig:supp-cumulants} use the full grid.}",
        r"\label{tab:supp-cumulants}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrr}",
        r" & $\gamma_1(1)$ & $\gamma_1(128)$ & $\gamma_2(1)$ & $\gamma_2(128)$ \\",
        r"\colrule",
    ])
    for row in exp14["results"]:
        first, last = row["standardized_cumulants"][0], row["standardized_cumulants"][-1]
        lines.append(
            f"{row['asset']} & {first['gamma1']:.2f} & {last['gamma1']:.2f} & "
            f"{first['gamma2']:.2f} & {last['gamma2']:.2f} \\\\"
        )
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", ""])

    lines.extend([
        r"\begin{table*}[t]",
        r"\caption{Sensitivity of the weekly counting-variance null. $R_{50}=F_{\rm obs}/\mathrm{median}(F_{\rm null})$ and $R_{98.75}=F_{\rm obs}/Q_{0.9875}(F_{\rm null})$. The empirical $p$ value includes the observed ordering. The last column is the fraction of candles fixed because their edge stratum contains one observation.}",
        r"\label{tab:supp-null}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{llrrrr}",
        r"asset & conditioning & $R_{50}$ & $R_{98.75}$ & $p_{\rm emp}$ & fixed fraction \\",
        r"\colrule",
    ])
    for row in exp14["results"]:
        for null in row["stratified_nulls"]:
            lines.append(
                f"{row['asset']} & {null['stratification']} & "
                f"{null['variance_ratio_to_null_median']:.2f} & "
                f"{null['variance_ratio_to_null_quantile']:.2f} & "
                f"{_pvalue(null['empirical_p_upper'])} & "
                f"{100 * null['singleton_fraction']:.2f}\\% \\\\"
            )
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}", ""])

    wavelets = {row["activo"]: row for row in exp11["reales"]}
    lines.extend([
        r"\begin{table}[t]",
        r"\caption{Global exponent $\nu$ over $T=8,\ldots,256$. The wavelet estimators act directly on signed flow: db1 removes a constant, db2 a linear trend, and db3 a quadratic trend.}",
        r"\label{tab:supp-wavelets}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrr}",
        r"asset & block variance & db1 & db2 & db3 \\",
        r"\colrule",
    ])
    for asset in ("BTC", "ETH", "BNB", "SOL"):
        row = wavelets[asset]
        lines.append(
            f"{asset} & {row['bloques']['nu_global']:.2f} & "
            f"{row['db1']['nu_global']:.2f} & {row['db2']['nu_global']:.2f} & "
            f"{row['db3']['nu_global']:.2f} \\\\"
        )
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", ""])

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
