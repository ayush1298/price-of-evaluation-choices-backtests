# Soft-regime-conditioned dynamic-model diagnostic

This document records a negative experiment so that it cannot be promoted into
an ICAIF claim by selective reporting. It evaluates the candidate configuration
in `configs/public_*_regime_conditioned_dynamic_graph.json` on the frozen public
ETF panels. It is a five-epoch, five-seed diagnostic, not the final 20-epoch
paper specification.

## Candidate

At every monthly decision date, four market features (realised volatility,
trailing market return, market drawdown, and cross-sectional dispersion) are
computed from data available at that close. An expanding standard scaler and
three-state K-Means model are fitted only through the previous monthly row.
Soft, distance-based state probabilities are then joined to each asset's node
features at the *same* decision date; no regime probability is forward-filled.
The existing causal dynamic graph Transformer, allocator, execution lag,
10 bps one-way costs, and 10% weight cap are otherwise unchanged.

The output is auditable in the corresponding `results/canonical/` run
directories: each run contains the probability panel, model-training dates,
targets, execution ledger, and input hashes.

## US fixed-ETF diagnostic

The US panel has 30 fixed ETFs, a 2020-01-31 outer signal boundary, and 1,627
realised daily observations through 2026-07-24. All five regime-conditioned
runs therefore have the exact same period as the regime-free dynamic model.

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Regime-free dynamic model, five-seed mean | 9.27% ± 1.39% | 14.45% ± 1.49% | 0.685 ± 0.065 | -26.72% ± 3.84% |
| Soft-regime-conditioned dynamic model, five-seed mean | 8.26% ± 1.04% | 14.17% ± 1.34% | 0.634 ± 0.079 | -24.82% ± 4.03% |
| Equal weight | 10.70% | 15.92% | 0.718 | -30.87% |

For seed 7, the conditional model's 21-session, 5,000-replication paired
moving-block comparison has an annualised mean-return difference of -0.01%
versus the regime-free model (`p=0.519`) and -1.13% versus equal weight
(`p=0.755`). These are not positive-performance results.

## India fixed-ETF diagnostic, date-matched

The India panel has 12 fixed ETFs in INR. The K-Means warm-up plus dynamic
sequence/label requirement permits the first conditioned target on 2020-09-30,
which executes on 2020-10-01. Every comparator was therefore repriced from
its immutable target stream over the exact 1,414 realised dates ending
2026-07-24. This avoids comparing returns from different market intervals.

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Regime-free dynamic model, five-seed mean | 20.80% ± 0.85% | 13.76% ± 0.65% | 1.445 ± 0.090 | -12.34% ± 0.24% |
| Soft-regime-conditioned dynamic model, five-seed mean | 20.50% ± 0.72% | 13.79% ± 0.69% | 1.425 ± 0.092 | -12.82% ± 0.49% |
| HMM mixture, five-seed mean | 21.74% ± 0.07% | 13.89% ± 0.00% | 1.486 ± 0.004 | -12.02% ± 0.00% |
| Equal weight | 21.42% | 13.29% | 1.528 | -11.91% |
| PCC hub-avoid + mean-CVaR | 21.61% | 13.89% | 1.478 | -11.84% |
| Partial-correlation hub-avoid + mean-CVaR | 22.01% | 12.68% | 1.633 | -12.45% |

As a seed-average diagnostic, the conditional model has an annualised daily
mean-return difference of -0.24% against the matched regime-free dynamic
model (`p=0.688`), -0.93% against PCC/CVaR (`p=0.904`), and -0.70% against
equal weight (`p=0.892`). Each is a 21-session, 5,000-replication paired
moving-block comparison. The seed average is an uncertainty summary, not an
additional tradable ensemble strategy.

## Decision and next model experiment

Do **not** use this direct-probability-concatenation model as the proposed
method. Retain it as a causal regime-conditioning ablation. The current
evidence says that the allocator/forecast stage, rather than simply the regime
feature, is the main bottleneck.

Any future candidate should keep the shared data, graph, and backtest contracts
but learn a small regime-conditioned forecast head (for example, a
probability-gated mixture of shared expert heads with a regularisation term)
and choose its complexity, forecast shrinkage, and turnover penalty strictly
on pre-2020 validation windows. It must be tested against the same simple
baselines and exact date-matched periods before being considered for the paper.

## Probability-gated expert-head development gate

The next candidate above was implemented as three small state-specific Gaussian
forecast heads. Their means and variances are mixed by the causal soft K-Means
probabilities at the final input date, including between-expert uncertainty in
the reported variance. This preserves the existing graph, Transformer,
allocator, cost model, and all leakage guards. The gated-head configuration is
`configs/public_us_etf_2010_regime_gated_experts_dynamic_graph.json`.

On the US development period that all three models can realise (2016-05-02 to
2019-12-31; 924 daily observations; seed 7; five epochs; 10 bps one-way
costs), it does not pass the pre-2020 gate:

| Variant | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Regime-free dynamic model | 9.94% | 9.75% | 1.021 | -13.88% |
| Direct probability concatenation | 8.64% | 10.48% | 0.843 | -14.31% |
| Probability-gated expert heads | 8.39% | 10.29% | 0.834 | -20.00% |

The gated model's paired 21-session moving-block bootstrap annualised
mean-return difference is -0.25% versus direct concatenation (`p=0.551`) and
-1.37% versus the regime-free model (`p=0.719`). It is therefore not advanced
to 2020+ or India evaluation. The implementation and frozen configurations are
retained for reproducibility, but the paper should not describe gated experts
as a selected method.
