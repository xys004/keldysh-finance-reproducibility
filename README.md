# Signed order flow as a transport current

This repository contains the reproducible code, archived analysis outputs, and
manuscript sources for the study of signed taker flow in cryptocurrency
markets. It is organised around two questions:

1. which response, noise, and counting observables are fixed by the measured
   second-order correlation function; and
2. which features remain outside a Gaussian response-field description and a
   calibrated noninteracting transport benchmark.

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
```

The scripts write JSON results to `output/`. Cached downloads are kept outside
version control. The manuscript figures can be regenerated from the archived
logs with:

```bash
python experiments/figuras_paper.py
```

To build the integrated long article:

```bash
python tools/build_combined_article.py
cd paper
pdflatex combined_article.tex
bibtex combined_article
pdflatex combined_article.tex
pdflatex combined_article.tex
```

## Scope of the results

The robust empirical results are superdiffusive counting-variance growth and a
large trade-size-normalised Fano factor. The relation between a power-law
correlation tail and the counting exponent is derived independently in the
response-field model. The finer drift of the local exponent remains an open
test because block estimators are sensitive to slow changes in the mean.

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

Use the repository citation metadata in `CITATION.cff`. Replace the repository
URL and DOI placeholders after the public release is registered.
