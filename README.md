# Temporal organization in signed order flow: reproducibility package

This repository contains the code, archived numerical outputs, figures, and
manuscript sources for the empirical study:

> Temporal Organization Amplifies Counting Fluctuations in Signed Order Flow

It is the release candidate for version 2.0.0. The existing Zenodo record for
version 1.2.0 remains an archival record of the earlier analysis; do not use
that version to reproduce the current manuscript. A new versioned release must
be published from the commit containing this package before submission.

## Main result and scope

The primary panel contains BTC, ETH, BNB, and SOL against USDT over four years.
XRP, ADA, DOGE, and AVAX form a post-hoc robustness panel and are not substituted
into the original decision rule. We analyse raw signed base volume,
`2 * tbBaseVolume - Volume`, and volume-normalised imbalance,
`(2 * tbBaseVolume - Volume) / Volume`.

Complete UTC weeks are conditioned on calendar quarter and hour of week. A
calendar-matched permutation preserves transfer sizes, strata, and one-point
statistics while destroying chronology. The observed weekly variance exceeds
this null under both flow definitions. The null-aligned finite-sample identity
then partitions the variance into size, sign-memory, and sign--size cross terms,
including the recentering correction required by the operational surrogate.

Within the excess over the size baseline, the corrected cross contribution
exceeds the sign-memory contribution in all eight primary point estimates. Six
paired eight-week moving-block bootstrap intervals exclude one half. This is an
exact allocation conditional on the declared null and reference convention; it
does not identify a causal interaction or a microscopic market mechanism.

The supplemental whole-week-shift diagnostic evaluates all 205 nonzero circular
weekly shifts. It finds the null-aligned cross term extreme in all eight series;
the global reference has one exception, raw BNB. Synthetic controls establish
sensitivity to strong weekly synchrony, not to every possible short-lag
dependence.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The test matrix checks the finite-sample recentering identities, the exhaustive
shift calculation, mutation detection in its independent verifier, and the
earlier causality and positive-control gates.

## Reproduce the submission analyses

The deposited JSON files in `output/` are the evidence used by the manuscript.
The public Binance candle cache is intentionally excluded. To rerun an analysis,
retrieve the same public candle snapshots described in `data/README.md` or use
the downloader in `src/keldysh_finance/flow.py`.

```bash
python experiments/exp15_flow_null_decomposition.py
python experiments/exp16_scaling_reassessment.py
python experiments/exp17_cumulant_memory_null.py
python experiments/exp24_null_recentring.py
python experiments/exp27_exhaustive_week_shift.py
python experiments/verify_exp27_exhaustive_week_shift.py
python experiments/render_supplement_tables.py
```

The independent verifier reloads the public source data and recomputes selected
shifts without importing the production shift routine. Its self-test reverses
the shift direction and must fail.

## Build the manuscript and supplement

From `paper/`:

```bash
latexmk -pdf combined_article.tex
latexmk -pdf paper3_supplement.tex
```

The canonical sources are `paper/combined_article.tex` and
`paper/paper3_supplement.tex`. The older financial Letter has been excluded
from this release: the separate structured-bath transport Letter has its own
solver and must be released independently.

## Build a portable release

```bash
python tools/build_release_archive.py
```

This creates `dist/keldysh-finance-reproducibility.zip`. Member names use
forward slashes, have one safe top-level prefix, and deterministic timestamps.

## Data provenance and license

No proprietary market feed is redistributed. The analysis uses public Binance
candle and trade archives; request parameters and source-cache SHA-256 hashes
are preserved in the deposited outputs. See `data/README.md` for details.

The code is MIT licensed. Citation metadata for the release candidate are in
`CITATION.cff`; a version-specific Zenodo DOI will be added only on publication.
