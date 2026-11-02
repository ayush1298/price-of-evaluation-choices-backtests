# ICAIF Experiment Protocol

## Purpose and non-claims

This protocol governs the canonical pipeline under `src/research/`. The
dashboard JSON files and legacy outputs are traceability artefacts only; none
of their Sharpe, drawdown, regime-label, or event-match figures are paper
results until reproduced by a canonical run.

The proposed contribution is not “a regime switch made after observing annual
returns.” It is a causal, cost-aware system that combines dynamic financial
graphs, probabilistic regime uncertainty, and constrained allocation. Every
component must improve a held-out result or remain an ablation.

## Locked decision timeline

At month-end `t`:

1. read the point-in-time eligible universe and all prices through the close
   of `t`;
2. construct PCC, shrinkage partial-correlation, and (where genuine OHLCV is
   available) seeded DeepCNL graph snapshots from historical windows only;
3. infer the regime using a scaler/model trained strictly before `t`;
4. create target weights; execute at the configured future close and deduct costs;
5. expose the return after `t` only as a later training label.

No scaler, graph edge, regime feature, hyperparameter, asset membership,
normalisation statistic, or label may use a later observation.

## Evaluation ladder

Every market and period must run the complete ladder below with identical
point-in-time universes, rebalance dates, dates of execution, portfolio caps,
and cost assumptions.

| Layer | Required comparisons |
| --- | --- |
| Allocation | equal weight, inverse volatility, long-only mean-CVaR with turnover penalty |
| Graph | no-graph allocation, PCC, shrinkage partial correlation, legacy DeepCNL where real OHLCV permits |
| Selection | all eligible assets, hub-follow, hub-avoid |
| Regime | no regime, causal expanding K-Means, causal Gaussian HMM hard routing, HMM probability mixture |
| Proposed model | dynamic graph Transformer with graph and temporal ablations, followed by the same mean-CVaR allocator |

Set `graph_layers=0` for the temporal-only ablation and `temporal_layers=0`
for the graph-only ablation. Both use the same input features, allocator,
backtest, and outer split as the full model; a configuration with neither stage
is rejected.

The dynamic model is evaluated using
`scripts/run_canonical_dynamic_graph_model.py`. It retrains from only completed
forward labels and records the last label date in
`model_training_diagnostics.csv` for every decision. Its predicted return
scores are sent through the same constrained allocator as non-neural methods;
an apparent gain therefore cannot be attributed merely to a different
backtester or a looser constraint set.

## Data and cross-market design

Use genuinely point-in-time constituent histories, including delisted firms,
and documented split/dividend-adjusted OHLCV. Keep local-currency results
separate from any USD-converted aggregate; an FX conversion rule, benchmark,
calendar, corporate-action treatment, and source hash are required for every
market.

Initial target universes are US large-cap equities, an India broad-market
universe, developed-Europe equities, and Japan equities. A market is included
only after it has the same membership and data-quality contract; it is not
acceptable to replace unavailable membership history with current constituents.

For each market, freeze the master tensor universe from the first model signal
date and mask later ineligible assets. This prevents tensor-shape changes from
silently dropping delistings while avoiding future asset identities in the
model input.

## Train, validation, and test protocol

Use a rolling-origin outer evaluation. A recommended first specification is a
five-year expanding training period, one-year validation period, and one-year
test period, advanced by one year. If a market has shorter usable history,
report the reduced design and do not pool it with the full-history results.

Within each outer training/validation span:

- select graph sparsity, HMM state count, model dimension, sequence length,
  CVaR aversion, and turnover penalty only on validation performance;
- use the selected configuration unchanged for that outer test year;
- run seeds 7, 17, 29, 43, and 59 for every stochastic graph/model;
- report median, interquartile range, and worst seed, not only the best seed;
- retain all candidate configurations and failed runs in the manifest ledger.

Before any confirmation/outer period is opened, commit a versioned validation
candidate ledger, audit every named manifest-backed validation run, and freeze
the selected configuration. The implementation and required artefacts are in
[`VALIDATION_CANDIDATE_LEDGER.md`](VALIDATION_CANDIDATE_LEDGER.md). An
incomplete or archival/post-hoc ledger is not eligible to select a candidate.

The 2010--2016 source panel is useful only as an initial corrected US
replication. It is too short to support broad hyperparameter search plus a
credible final held-out study; the extended data gate in
`RESEARCH_MIGRATION.md` is mandatory.

## Reported metrics and robustness tests

Report daily net return, annualised return/volatility, Sharpe, Sortino,
Calmar, daily-NAV maximum drawdown, 95% daily CVaR, turnover, realised costs,
average holdings, and time in each regime. Include gross and net results.

Run all strategies at at least 0, 5, 10, 25, and 50 basis points one-way costs,
plus a liquidity/capacity screen when volume data is licensed. Include
portfolio caps of 2%, 5%, and 10%; monthly and quarterly rebalancing; and
graph lookbacks of 63, 126, and 252 sessions.

For inference, use a stationary/block bootstrap for the return differential,
paired across strategies by date. Apply White's Reality Check or Hansen's SPA
to the complete model-selection set, and report deflated Sharpe / probability
of backtest overfitting diagnostics. All p-values and confidence intervals
must be produced from the out-of-sample return series, never from annual
summary values.

`scripts/compare_canonical_runs.py` implements the paired moving-block
bootstrap comparison for two manifest-backed daily-NAV runs. The evaluation
package also exposes a deflated-Sharpe diagnostic; provide it with the complete
set of validation trial Sharpes, not only selected configurations.
`scripts/white_reality_check_canonical_runs.py` implements the matching
moving-block White Reality Check across the entire candidate ledger. It refuses
an incomplete ledger or a candidate with a different data/split/backtest
context; usage and scope are documented in
[`MULTIPLICITY_AWARE_INFERENCE.md`](MULTIPLICITY_AWARE_INFERENCE.md).

## Recent-work implications incorporated into this project

- [Korangi, Mues & Bravo (2024), *Large-scale Time-Varying Portfolio
  Optimisation using Graph Attention Networks*](https://arxiv.org/abs/2407.15532)
  motivates a graph-neural portfolio comparator that preserves firms at risk
  of leaving the universe. The canonical dynamic graph Transformer now keeps
  an explicit master universe and eligibility mask. The pipeline now includes
  a matched causal edge-masked graph-attention baseline; its first development
  result is recorded in `GRAPH_ATTENTION_BASELINE_RESULTS.md` and is not a
  selected method.
- [Majumder, Ji & Neerchal (2024), *Optimal Stock Portfolio Selection with a
  Multivariate Hidden Markov Model*](https://arxiv.org/abs/2406.02297)
  highlights sector heterogeneity rather than a single retrospective
  bull/bear label. The current Gaussian HMM is a transparent common-regime
  baseline; a sector-conditioned/linked regime model is a planned ablation,
  not yet a claimed contribution.
- [Jensen, Kelly, Malamud & Pedersen (2024), *Machine Learning and the
  Implementable Efficient Frontier*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4187217)
  reinforces that an ML forecast must be judged after realistic trading
  implementation. The pipeline therefore uses a shared cost-aware backtest,
  turnover penalty, and constrained mean-CVaR allocation for every method.
- [Barroso, Saxena & Wang (2025), *Calm Your Portfolio*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5046369)
  motivates explicit forecast/weight stabilisation. Sensitivity analyses must
  include turnover penalties and volatility/weight caps; a no-trade band is a
  planned allocator ablation.
- [Boukardagha (2026), *Explainable Regime Aware
  Investing*](https://arxiv.org/abs/2603.04441) is a recent preprint, not a
  benchmark claim. Its emphasis on causal state identity and probability-based
  allocation supports the current soft HMM routing; adaptive state count and
  Wasserstein identity tracking remain future work.

## Conditions for a paper table

An ICAIF table may be exported only when each run directory includes its input
hashes, configuration, code revision, target weights, executions, daily NAV,
holdings, graph/regime diagnostics, and seed. The result must survive the
specified cost, cap, seed, graph, and market robustness checks. Any row that
uses a static current universe, synthetic OHLCV, a post-hoc regime label, or a
legacy output file is excluded.

Use `scripts/summarize_canonical_runs.py` to build a draft table. It reads only
directories that contain both a canonical manifest and daily-performance
metrics; graph-only smoke runs and all `results/legacy` material are excluded.

### Mechanical paper-release gate

Reproducibility is necessary but not sufficient for paper status.  Every new
canonical manifest is classified as non-paper by default, even if its run
directory has all normal outputs.  To make a result eligible for a paper table,
its JSON configuration must explicitly set `paper_eligible: true` and supply a
`paper_evidence` object with all of the following set to `true`:

```json
{
  "universe_contract_verified": true,
  "price_adjustment_verified": true,
  "evaluation_precommitted": true,
  "independent_confirmation": true,
  "robustness_matrix_completed": true
}
```

The manifest records the resolved status and any missing conditions.  The
standard summarizer exports only explicitly paper-ready runs; it fails rather
than silently treating an unclassified run as publishable.  Its
`--include-nonpaper` switch is solely for an internal diagnostic table and
must never be used as a manuscript-table export.  This gate is an auditable
release check, not a substitute for inspecting the declared data provenance.
