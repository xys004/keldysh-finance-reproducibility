"""Render tables used only by the focused Letter supplement."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
TARGET = ROOT / "paper" / "generated_letter_tables.tex"


def _load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _boot(result: dict, block: int) -> dict:
    return next(
        row for row in result["paired_weekly_block_bootstrap"]
        if row["block_length_weeks"] == block
    )


def _close(lines: list[str]) -> None:
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}", ""])


def main() -> None:
    exp15 = _load("exp15_flow_null_decomposition.json")
    exp17 = _load("exp17_sign_size_coupling.json")
    lines: list[str] = []

    lines.extend([
        r"\begin{table*}[!htbp]",
        r"\caption{Exact quarterly attribution. $A=V_{\rm obs}/V_0$; $f_s$ and $f_{sm}$ divide $V_{\rm obs}-V_0$. Intervals are paired eight-week circular moving-block intervals for $f_{sm}$. The first four assets are primary.}",
        r"\label{tab:exact-attribution}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{llrrrrr}",
        r"asset & flow & $A$ & $f_s$ & $f_{sm}$ & 95\% interval & $P(f_{sm}>1/2)$ \\",
        r"\colrule",
    ])
    for index, row in enumerate(exp17["results"]):
        if index == 4:
            lines.append(r"\colrule")
        for flow in ("raw", "normalized"):
            result = row["quarterly"][flow]
            ci = _boot(result, 8)
            lines.append(
                f"{row['asset']} & {flow} & "
                f"{result['amplification_over_marginal']:.2f} & "
                f"{result['sign_share_of_excess']:.2f} & "
                f"{result['coupling_share_of_excess']:.2f} & "
                f"[{ci['coupling_share_ci025']:.2f},"
                f"{ci['coupling_share_ci975']:.2f}] & "
                f"{ci['probability_coupling_share_gt_half']:.3f} " + r"\\"
            )
    _close(lines)

    lines.extend([
        r"\begin{table*}[!htbp]",
        r"\caption{Block-length sensitivity of the primary raw-flow coupling share. Each entry is the paired moving-block 95\% interval.}",
        r"\label{tab:block-sensitivity}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrrr}",
        r"asset & point & 4 weeks & 8 weeks & 13 weeks & 26 weeks \\",
        r"\colrule",
    ])
    for row in exp17["results"][:4]:
        result = row["quarterly"]["raw"]
        intervals = []
        for block in (4, 8, 13, 26):
            ci = _boot(result, block)
            intervals.append(
                f"[{ci['coupling_share_ci025']:.2f},{ci['coupling_share_ci975']:.2f}]"
            )
        lines.append(
            f"{row['asset']} & {result['coupling_share_of_excess']:.2f} & "
            + " & ".join(intervals) + r" \\"
        )
    _close(lines)

    lines.extend([
        r"\begin{table*}[!htbp]",
        r"\caption{Conditioning-scale sensitivity in the primary panel. Two-point biweekly strata force equal residual magnitudes after demeaning, so their zero coupling remainder is a degeneracy, not evidence of absence.}",
        r"\label{tab:conditioning-sensitivity}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lllrrr}",
        r"asset & flow & conditioning & $f_{sm}$ & 95\% interval & status \\",
        r"\colrule",
    ])
    for row in exp17["results"][:4]:
        for flow in ("raw", "normalized"):
            results = [row["quarterly"][flow]]
            results.extend(row["stratification_sensitivity"][flow])
            for result in results:
                ci = _boot(result, 8)
                status = "degenerate" if result["stratification"] == "biweekly" else "identified"
                lines.append(
                    f"{row['asset']} & {flow} & {result['stratification']} & "
                    f"{result['coupling_share_of_excess']:.2f} & "
                    f"[{ci['coupling_share_ci025']:.2f},"
                    f"{ci['coupling_share_ci975']:.2f}] & {status} " + r"\\"
                )
    _close(lines)

    lines.extend([
        r"\begin{table*}[!htbp]",
        r"\caption{Legacy non-additive permutation diagnostics. $R_J$, $R_S$, and $R_M$ divide observed variance by the medians of joint-order, sign-order, and magnitude-order nulls. The latter two break contemporaneous sign--magnitude pairing and are controls, not variance components.}",
        r"\label{tab:permutation-controls}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{llrrr}",
        r"asset & flow & $R_J$ & $R_S$ & $R_M$ \\",
        r"\colrule",
    ])
    for index, row in enumerate(exp15["results"]):
        if index == 4:
            lines.append(r"\colrule")
        for mode in row["flow_definitions"]:
            values = {item["null_mode"]: item for item in mode["primary_quarterly"]}
            lines.append(
                f"{row['asset']} & {mode['flow_mode']} & "
                f"{values['joint']['variance_ratio_to_null_median']:.2f} & "
                f"{values['sign']['variance_ratio_to_null_median']:.2f} & "
                f"{values['magnitude']['variance_ratio_to_null_median']:.2f} " + r"\\"
            )
    _close(lines)

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
