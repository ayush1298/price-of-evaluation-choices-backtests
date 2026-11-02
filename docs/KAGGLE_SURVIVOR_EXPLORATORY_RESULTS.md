# Kaggle Current-S&P Survivor Panel: Baseline Results

## Scope

These are explicitly exploratory results from the current-S&P survivor panel
documented in [the runbook](KAGGLE_SURVIVOR_EXPLORATORY_RUNBOOK.md).  They are
useful for model development only.  They must not be included in a paper
headline table or described as a historical S&P 500 result, because current
constituents replace the point-in-time universe and the source's corporate
action convention is not independently verified.

Protocol: 88 sector-balanced current survivors, 2010-01-04 to 2025-12-31,
monthly rebalancing, one-session execution lag, 10 bps one-way transaction
cost, 5% maximum weight.  Development selection uses 2015-01-01 through
2019-12-31; the frozen outer period is 2020-01-01 through 2025-12-31.  Price
history before each signal remains available for the required 252-session
estimation window.

## Development ladder (2015--2019)

| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Daily CVaR 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 13.48% | 12.82% | 1.051 | -17.18% | -2.05% |
| Inverse volatility | 13.10% | 12.15% | 1.074 | -15.82% | -1.95% |
| Hierarchical risk parity (HRP) | 12.67% | 11.66% | 1.082 | -14.08% | -1.86% |
| PCC graph, hub avoidance, mean-CVaR (44 names) | 12.89% | 11.79% | 1.087 | -12.15% | -1.80% |
| Partial-correlation graph, hub avoidance, mean-CVaR (44 names) | 11.82% | 11.16% | 1.058 | -12.05% | -1.70% |

The nominal PCC/CVaR lead over HRP is not evidence of an edge: its paired
21-session moving-block bootstrap has annualized excess return 0.21%, a 95%
interval from -2.58% to 2.97%, and one-sided p=0.470.  Across all five
development trials, its deflated-Sharpe probability is 0.499.  The marginal
gain therefore fails the pre-specified robustness bar, and the simpler HRP
benchmark was frozen as the selected development strategy.

## Frozen outer result (2020--2025)

| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Daily CVaR 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 11.16% | 20.85% | 0.612 | -38.51% | -3.11% |
| Frozen HRP | 6.42% | 18.07% | 0.435 | -34.91% | -2.69% |

HRP reduced volatility and tail loss but did not compensate through returns.
Versus equal weight, the paired moving-block bootstrap estimated -4.90%
annualized excess return for HRP, with a 95% interval of -9.31% to -0.15% and
a one-sided p=0.984 for the hypothesis that HRP is superior.  Thus the
development selection did not generalize to this outer period.

## Decision

No graph-selection, CVaR, or current regime/allocation method earns promotion
from this panel.  Do not tune a neural model to reverse this outcome.  A new
model may use this panel for training diagnostics only if it is evaluated
against the same fixed baseline protocol and then confirmed unchanged on an
independently sourced, point-in-time universe with documented adjustments.

The results are stored in ignored `results/canonical/exploratory-kaggle-*`
directories; their manifests carry `paper_eligible=false`, so normal paper
table collection excludes them by construction.
