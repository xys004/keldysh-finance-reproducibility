# Signed order flow as a transport current

This repository contains the reproducible code, archived analysis outputs, and
manuscript sources for the study of signed taker flow in cryptocurrency
markets. It is organised around two questions:

1. which response, noise, and counting observables are fixed by the measured
   second-order correlation function; and
2. how much counting variance remains after preserving transfer sizes,
   activity, seasonality, and quarterly drift while destroying temporal order.

The repository contains no proprietary market feed. Raw candles are not
redistributed; the analysis code retrieves the required public Binance klines,
or can read a locally cached CSV with the same columns. The archived JSON
files and figures are the exact outputs used by the manuscript versions in
`paper/`.

## Reproduce the published checks

```bash
python -m pip install numpy pandas scipy matplotlib pytest
python -m pytest tests -q
python experiments/exp06_conteo_flujo.py
python experiments/exp07_reloj_quench.py
python experiments/exp08_floquet.py
python experiments/exp09_msrjd_orden2.py
python experiments/exp11_haar_crossover.py
python experiments/exp12_fano_memory_null.py
python experiments/exp13_trade_level_validation.py
python experiments/exp14_referee_robustness.py
python experiments/plot_fano_memory_null.py
python experiments/plot_referee_robustness.py
python experiments/render_supplement_tables.py
```

The scripts write JSON results to `output/`. Cached downloads are kept outside
version control. The manuscript figures can be regenerated from the archived
logs with:

```bash
python experiments/figuras_paper.py
```

To build the integrated PRE article:

```bash
cd paper
pdflatex combined_article.tex
bibtex combined_article
pdflatex combined_article.tex
pdflatex combined_article.tex
```

To build the PRL Letter and its Supplemental Material from a clean checkout:

```bash
cd paper
pdflatex paper3_letter.tex
bibtex paper3_letter
pdflatex paper3_letter.tex
pdflatex paper3_letter.tex
pdflatex paper3_supplement.tex
bibtex paper3_supplement
pdflatex paper3_supplement.tex
pdflatex paper3_supplement.tex
```

The concise PRL manuscript is `paper/paper3_letter.tex`, with
`paper/paper3_supplement.tex` as its Supplemental Material. The earlier split
measurement/model drafts have been retired; `combined_article.tex` is the
canonical integrated source.

## Scope of the results

The robust empirical result is memory-amplified counting noise: at a weekly
horizon, the observed/null variance ratio is 2.20--3.37 across BTC, ETH, BNB,
and SOL. The stratified null preserves the empirical flow-size marginal, its
pairing with trade activity, hour-of-week seasonality, and quarterly scale
changes. The ratio is independent of the transfer-size normalisation.

Monthly conditioning preserves a ratio above two for every asset. With
biweekly strata the ratio falls below two for BTC and BNB, although all four
observations remain above the 98.75th percentile of their conditioned nulls.
This is reported as scale sensitivity, not partition invariance. Standardised
third and fourth cumulants likewise show no common Gaussian crossover scale.

The absolute Fano level remains descriptive because candle data contain a
median of hourly mean sizes, not the median individual trade size. A calibration
on 14.8 million public trades quantifies this difference. The relation between a
power-law correlation tail and the superdiffusive counting exponent is derived
independently in the response-field model. The finer drift of the local exponent
remains an open test because block estimators are sensitive to slow changes in
the mean.

The public data statement and the manuscript's limits should be read before
interpreting any output as a microscopic market model.

## Layout

- `src/keldysh_finance/` — analysis library;
- `experiments/` — reproducible entry points for the reported results;
- `output/` — archived JSON results and figures;
- `paper/` — manuscript sources and bibliography;
- `data/` — data provenance and download notes;
- `tests/` — numerical and causal-consistency tests.

## Citation

Use the repository citation metadata in `CITATION.cff` or cite the archived
release: https://doi.org/10.5281/zenodo.21927599.
