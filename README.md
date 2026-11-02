# The Price of Evaluation Choices in Portfolio Backtests

**Ayush Sunil Munot**, Indian Institute of Technology Kharagpur, 2026.

Accepted at the ACM International Conference on AI in Finance (ICAIF '26, Milan, November 2026);
withdrawn by the author before publication. Not published elsewhere. This repository holds the
author's version of the paper, its LaTeX source, and the code that produced every number in it.

**Paper:** [`paper/The_Price_of_Evaluation_Choices_in_Portfolio_Backtests.pdf`](paper/The_Price_of_Evaluation_Choices_in_Portfolio_Backtests.pdf)

## Abstract

A backtest reports a number, but that number depends on choices the reader never sees: which
baseline was used, over which dates, at which trading cost, and at which random seed. The paper
measures what those choices are worth. Nine regime-aware, graph-based and neural portfolio methods
are built inside one causal, cost-aware pipeline with a shared allocator and backtester; none beats
a simple baseline out of sample. That gives a test bed with no detected edge, so any performance
still visible in the runs comes from how they were reported. Across 345 manifest-backed runs on
four fixed-instrument panels and two markets, the data, allocator, cost model and execution rule
are held fixed and one reporting choice is changed at a time: trading costs, the development/held-out
boundary, date alignment, seed selection, benchmark selection, and multiplicity correction. A
pre-registered prediction of failure is then tested on a sealed panel sharing no instrument with the
development data, and the trading-cost result repeats out of sample.

## What is in the repository

| Path | Contents |
| --- | --- |
| `paper/` | The PDF and its LaTeX source (`paper/source/`: `main.tex`, bibliography, tables, figures, vendored `acmart.cls`). Build with `latexmk -pdf main.tex`. |
| `src/research/` | The point-in-time pipeline: data contracts and loaders, graph construction, regime models (HMM and soft-regime risk), allocation (equal weight, inverse volatility, hierarchical risk parity, mean-CVaR with turnover penalty), the causal backtester and walk-forward harness, and multiplicity-aware evaluation (paired moving-block bootstrap, White's Reality Check, deflated Sharpe). |
| `scripts/` | Run scripts for the canonical experiments, confirmation inference, the protocol-inflation table, Reality Check, and the paper's figures and tables. |
| `configs/` | Experiment configurations, data contracts, and the frozen validation-candidate ledger. |
| `results/` | Canonical and confirmation run outputs (manifests, metrics, inference JSON) and `protocol_inflation_table_v1.json`. |
| `docs/` | The experiment charter and its amendments, the pre-registered confirmation protocol, per-method result notes, and the multiplicity-aware inference note. |
| `tests/` | `test_research_pipeline.py`: contract tests for the causal barrier, backtester and walk-forward harness. |

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests/
# data retrieval and the canonical runs: see docs/PUBLIC_ETF_EXPERIMENT_RUNBOOK.md
# and docs/EXPANDED_US_ETF_EXPERIMENT_RUNBOOK.md for the exact commands per panel
python scripts/run_canonical_baseline.py --config configs/canonical_us_monthly.json
python scripts/run_confirmation_inference.py          # sealed-panel confirmation (pre-registered)
python scripts/white_reality_check_canonical_runs.py  # multiplicity-aware inference
python scripts/build_protocol_inflation_table.py      # Table 3 of the paper
python scripts/build_paper_tables.py && python scripts/build_paper_figures.py
```

`docs/ICAIF_EXPERIMENT_PROTOCOL.md` is the locked decision timeline every run follows;
`docs/ICAIF_CHARTER_AMENDMENT_V3_CONFIRMATION.md` is the pre-registration of the sealed-panel test.

Prices come from Yahoo Finance through `yfinance`; every run records the retrieval timestamp,
library version and a response hash in its manifest, so a later revision of the adjusted prices is
detected rather than silently absorbed. Bitwise reproduction of the confirmation cycle therefore
holds against the recorded inputs; a fresh download may differ at the level Yahoo revises.

## Scope

The magnitudes in the paper belong to these panels, strategies and windows. They are measurements
of reporting conventions, not a correction factor for anyone else's results, and the repository
contains no investable result.

## Citation

```bibtex
@unpublished{munot2026price,
  author = {Munot, Ayush Sunil},
  title  = {The Price of Evaluation Choices in Portfolio Backtests},
  year   = {2026},
  note   = {Accepted at ACM ICAIF 2026; withdrawn before publication. Author's version.},
  url    = {https://github.com/ayush1298/price-of-evaluation-choices-backtests}
}
```

## License

MIT. See `LICENSE`. `paper/source/acmart.cls` and `ACM-Reference-Format.bst` are the ACM
template files and keep their own licence (LPPL).
