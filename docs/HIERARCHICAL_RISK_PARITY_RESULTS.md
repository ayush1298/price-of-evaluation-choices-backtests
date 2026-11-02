# Hierarchical Risk Parity allocation benchmark

This is a conventional allocation benchmark, not a new proposed neural model.
It is added because the original mean--CVaR allocator depends on an expected
return estimate and can be unstable in a short monthly decision sample.
Hierarchical Risk Parity (HRP) changes only the allocation stage: it preserves
the fixed point-in-time ETF universe, historical-return lookback, graph
selection, monthly signal timing, one-session execution lag, 10 bps one-way
costs, and the existing leakage-safe backtester.

## Implementation contract

`hierarchical_risk_parity_weights` uses only return observations available at
the signal date. It computes correlation distance
`sqrt((1 - correlation) / 2)`, performs deterministic single-linkage
clustering with optimal leaf ordering, and allocates recursively between the
two sides of each cluster. A cluster's risk is measured as the variance of its
inverse-variance portfolio. The final allocation is long only, fully invested,
and respects the existing per-asset cap. There is no forward covariance,
future membership, or expected-return input.

The method is exposed in the canonical baseline runner as
`--allocator hierarchical_risk_parity`, so it can be paired with any current
graph/selection choice without duplicating backtest code.

## US 30-ETF development and outer results

The fair allocation comparison uses the same PCC graph with `selector=all`;
with every asset selected, the graph is inert and the comparison isolates the
allocator. All three runs use the public-ETF configuration's 10% cap. The
development window is bounded at 2019-12-31 (1,489 realised daily
observations), while the chronological outer window begins at the 2020-01-31
signal boundary (1,627 observations). All figures are net of 10 bps one-way
costs.

| Window | Allocator | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Equal weight | 7.84% | 10.19% | 0.792 | -16.08% |
| Development | Inverse volatility | 7.51% | 8.37% | 0.907 | -13.72% |
| Development | HRP | 6.59% | 7.31% | 0.910 | -13.05% |
| Outer | Equal weight | 10.70% | 15.92% | 0.718 | -30.87% |
| Outer | Inverse volatility | 8.86% | 13.72% | 0.688 | -27.30% |
| Outer | HRP | 8.14% | 12.05% | 0.710 | -24.10% |

HRP's development Sharpe is only 0.003 above inverse-volatility, while its
annualised mean return is 0.94% lower. A 5,000-replication, 21-session paired
moving-block bootstrap gives a one-sided mean-return p-value of 0.965. On the
outer period, HRP is below equal weight by 0.008 Sharpe and 2.88% annualised
mean return (one-sided p-value 0.977). It is therefore not a selected US
allocator.

## Wider US and India robustness results

The wider 50-ETF panel uses the same pre-2020 development / 2020+ outer split.
India is an independent public ETF market evaluated only on its already
exploratory 2020+ chronological period; its 10% cap makes equal-weight the
natural close comparator.

| Panel and window | Equal weight Sharpe | Inverse-volatility Sharpe | HRP Sharpe | Strongest existing baseline |
| --- | ---: | ---: | ---: | ---: |
| 50-ETF US development | 0.663 | 0.891 | 0.729 | PCC hub-avoid + mean-CVaR: 1.347 |
| 50-ETF US outer | 0.760 | 0.776 | 0.822 | n/a (development choice is binding) |
| India ETF outer | 1.295 | 1.294 | 1.300 | PCC hub-avoid + mean-CVaR: 1.313 |

The 50-ETF outer HRP result has a positive 1.06% annualised mean-return
differential over inverse-volatility (one-sided paired block-bootstrap
`p=0.018`), but its development Sharpe is far lower. It consequently fails the
development selection rule: selecting it after observing this outer result
would be test leakage. The India result is also below the existing PCC/CVaR
baseline. HRP remains in the experiment matrix as a transparent allocation
robustness check, but not as the paper's claimed improvement.

This negative result is useful: a stronger allocator alone does not solve the
main research problem. A proposed regime-aware method must still demonstrate
a development-selected, cross-market, factor-adjusted improvement over these
conventional allocation controls.
