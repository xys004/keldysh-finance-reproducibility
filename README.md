# Sign memory and counting noise in market order flow

This repository contains the code, archived numerical outputs, figures, and
manuscript sources for two related articles:

- a concise Letter on the attribution of memory-amplified counting variance;
- a pedagogical PRE manuscript translating between market microstructure and
  nonequilibrium transport.

The software release is archived at
[Zenodo](https://doi.org/10.5281/zenodo.21927599).

## Main result and scope

The primary panel contains BTC, ETH, BNB, and SOL against USDT over four years.
XRP, ADA, DOGE, and AVAX form a post-hoc asset-extension panel and are not
substituted into the original decision rule.

Two signed-flow observables are analysed in parallel:

- raw signed base volume, **2*tbBase-Volume**;
- volume-normalised imbalance, **(2*tbBase-Volume)/Volume**.

Complete UTC weeks are compared with three permutation nulls conditioned on
calendar period and hour of week:

- **joint**: destroys the complete signed-flow ordering;
- **sign**: preserves the magnitude path and destroys sign ordering;
- **magnitude**: preserves the sign path and destroys magnitude ordering.

The sign and magnitude nulls intentionally break contemporaneous
sign--magnitude pairing. They are attribution diagnostics, not structural
order-book models. Weekly moving-block bootstrap intervals are reported for
4-, 8-, 13-, and 26-week blocks.

The original predeclared gate required the observed raw-flow variance to exceed
twice the 98.75th null percentile in all four primary assets. It failed because
BNB did not pass. The normalised-flow analysis is reported as robustness, not
as a replacement criterion.

The scaling analysis is also deliberately limited. The autocorrelation and
accumulated-flow variance are linked by an exact second-order identity and are
not independent evidence. A single power law fails in all eight primary
asset--resolution series; the two-regime construction is only a consistency
band and fails for BNB. The archived effective-temperature diagnostic is not
claimed as a result, and the exploratory Floquet analysis is omitted from both
canonical manuscripts.

## Install and test

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q
~~~

The test matrix runs on Linux and Windows. It includes exact fractional
Gaussian-noise controls and verifies that the release ZIP contains only safe
POSIX paths.

## Reproduce the major-revision analyses

~~~bash
python experiments/exp10_gates_estimador.py
python experiments/exp14_referee_robustness.py
python experiments/exp15_flow_null_decomposition.py
python experiments/exp16_scaling_reassessment.py
python experiments/plot_referee_robustness.py
python experiments/plot_flow_null_decomposition.py
python experiments/plot_scaling_reassessment.py
python experiments/render_supplement_tables.py
~~~

The full historical analysis chain is preserved in **experiments/exp01_*.py**
through **experiments/exp16_*.py**. Deposited JSON files in **output/** are the
versioned evidence used by the manuscripts. Exploratory outputs remain
available for transparency even when the associated claims were withdrawn.

## Build the manuscripts

From **paper/**:

~~~bash
pdflatex paper3_letter.tex
bibtex paper3_letter
pdflatex paper3_letter.tex
pdflatex paper3_letter.tex

pdflatex paper3_supplement.tex
bibtex paper3_supplement
pdflatex paper3_supplement.tex
pdflatex paper3_supplement.tex

pdflatex combined_article.tex
bibtex combined_article
pdflatex combined_article.tex
pdflatex combined_article.tex
~~~

The canonical sources are:

- **paper/paper3_letter.tex**
- **paper/paper3_supplement.tex**
- **paper/combined_article.tex**

Compiled review copies are deposited under **output/pdf/**.

The earlier split measurement/model drafts are retired.

## Build a portable release

~~~bash
python tools/build_release_archive.py
~~~

This creates **dist/keldysh-finance-reproducibility.zip**. Member names use
forward slashes, share one safe top-level prefix, and have deterministic
timestamps so that archives are portable across Windows, Linux, and macOS.

## Data provenance

No proprietary feed is redistributed. The analysis retrieves public Binance
klines and records source-cache SHA-256 hashes in the deposited output. Raw
caches are excluded from version control. Experiment 13 additionally uses one
complete UTC day of public Binance Vision trade archives for BTCUSDT and
ETHUSDT to calibrate the transfer-size normalisation.

## Citation and license

Citation metadata are in **CITATION.cff**. Code is released under the MIT
License. Manuscript text and figures should be cited through the associated
articles and Zenodo record.
