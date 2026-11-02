# HMM uncertainty-aware conditional-risk results

## Status

**Not promoted.**  This is a causal, manifest-backed development experiment on
the already-seen public US ETF panel.  It does not use the independent
confirmation universe.  The method failed the precommitted development gate,
so running it on that panel now would be an avoidable selection error.

## Candidate

At each monthly close, an expanding Gaussian HMM is fit only on prior monthly
market observations.  The current soft posterior blends covariance estimates
conditioned on the previously observed state probabilities.  A normalized
posterior entropy score determines how much of the conditional estimate is
trusted: a diffuse posterior continuously reverts to unconditional covariance
risk rather than triggering a hard allocation switch.  The model records the
HMM `training_end`, state entropy/confidence, effective state observations,
fallback states, target weights, executions, daily NAV, and costs for every
decision.

The shared, 10%-capped HRP allocator, one-session execution lag, monthly
rebalance, and 10 bps one-way cost are unchanged.  The implementation is
`scripts/run_canonical_hmm_regime_risk.py`; its input data must first pass a
versioned data contract.

## Development selection grid

The fixed development window is 2014-01-31 through 2019-12-31 (1,489 realised
daily observations).  The registered grid combines 2/3/4 HMM states with
63/126/252-session covariance histories, an entropy-confidence floor of 0.10,
and no turnover band.  Values are net of cost.

| HMM states | Lookback | Annualised return | Annualised volatility | Sharpe | Maximum drawdown | 95% daily CVaR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 63 | 6.75% | 7.40% | 0.920 | -13.06% | -1.10% |
| 2 | 126 | 6.62% | 7.40% | 0.903 | -12.92% | -1.10% |
| 2 | 252 | **7.06%** | 7.38% | **0.961** | -12.47% | -1.09% |
| 3 | 63 | 6.88% | 7.39% | 0.937 | -12.45% | -1.09% |
| 3 | 126 | 6.89% | 7.39% | 0.939 | -12.27% | -1.09% |
| 3 | 252 | 6.71% | 7.33% | 0.923 | -12.46% | -1.09% |
| 4 | 63 | 6.75% | 7.38% | 0.923 | -13.29% | -1.09% |
| 4 | 126 | 6.75% | 7.44% | 0.915 | -13.39% | -1.11% |
| 4 | 252 | 6.63% | 7.41% | 0.903 | -12.97% | -1.10% |

The nominal development maximum is two states and 252 sessions.  Its 7.06%
annualised return and 0.961 Sharpe are only marginally higher than the
otherwise matched 756-session unconditional HRP control (6.94%, 0.956).  A
5,000-replication paired 21-session moving-block bootstrap gives:

| Strategy minus control | Annualised mean-return difference | Sharpe difference | 95% bootstrap interval (daily) | One-sided p-value |
| --- | ---: | ---: | ---: | ---: |
| HMM (2, 252) − HRP (252) | 0.11% | 0.013 | [-0.00065%, 0.00159%] | 0.209 |
| HMM (2, 252) − HRP (756) | 0.11% | 0.005 | [-0.00186%, 0.00270%] | 0.359 |

Neither comparison supports a positive net return advantage.  The HMM result
also has a slightly worse maximum drawdown and daily CVaR than the stronger
756-session HRP control, so it fails the charter's risk-consistency condition.

## Turnover-band check

The selected 2-state/252-session candidate was run with the predeclared
10% and 25% one-way turnover bands.  These values were not used to choose a
test-period configuration.

| Band | Sharpe | Annualised return | Maximum drawdown | 95% daily CVaR | Mean executed turnover | Skipped decisions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | **0.961** | 7.06% | -12.47% | -1.09% | 8.28% | 0 / 72 |
| 10% | 0.944 | 6.94% | -12.50% | -1.10% | 4.50% | 51 / 72 |
| 25% | 0.909 | 7.23% | -12.78% | -1.23% | 1.41% | 70 / 72 |

The 10% band materially lowers turnover but reduces Sharpe; the 25% band
reduces turnover further while worsening both risk-adjusted and tail metrics.
Neither is a performance solution.

## Decision

Keep the HMM uncertainty fallback as a transparent ablation and retain its
code/tests.  Do not advance it to the untouched public confirmation universe,
India, a manuscript table, or an outperformance claim.  The next valid model
change requires an amended charter and a genuinely independent confirmation
route; simply varying HMM/graph/neural choices again on the same development
panel would be model mining.

### Artefact-audit update

The archival candidate-ledger audit initially found only eight of the nine
manifest-backed grid artefacts; the `hmm=3`, `lookback=252` run directory was
absent. It was regenerated on 6 August 2026 from the unchanged frozen config,
data contract, and inputs. It reproduced the table row above exactly — Sharpe
0.9228, 6.71% annualised return, -12.46% drawdown, 1,489 observations — at a
later git revision, which is itself a useful reproducibility datum. The ledger
now audits 9/9 complete.

### Search-family (multiplicity-aware) result

With the family complete, White's Reality Check can be run over all nine
candidates against the matched 756-session unconditional HRP control:

| Comparison | p-value |
| --- | ---: |
| Paired bootstrap, best cell only (HMM 2/252 − HRP 756) | 0.359 |
| **White Reality Check, all nine searched cells** | **0.680** |

The best cell's 0.111% annualised excess return over the control does not
survive correction for the search that produced it. This strengthens rather
than changes the decision below: the candidate was already rejected on the
single-comparison test, and it fails more decisively once multiplicity is
priced in.

The archival ledger still explicitly forbids selection, and the HMM candidate
remains rejected.
