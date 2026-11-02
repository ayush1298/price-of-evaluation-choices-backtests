# Group-Aware HMM Conditional-Risk Diagnostic

## Status

**Not promoted.** This is the sole development diagnostic authorised by
[`ICAIF_CHARTER_AMENDMENT_V2_GROUP_AWARE_RISK.md`](ICAIF_CHARTER_AMENDMENT_V2_GROUP_AWARE_RISK.md).
It uses the already-seen public US ETF development panel and does not open the
alternative ETF confirmation universe.

## Fixed hypothesis

The model retains soft state-conditioned covariance only within five ex-ante
ETF groups—US core equity, US sector equity, international equity, fixed
income, and diversifiers—and replaces cross-group conditional covariance with
unconditional covariance. A positive-semidefinite projection is applied only
when this block construction introduces a negative eigenvalue. All HMM,
lookback, cost, execution, cap, and HRP allocation settings match the prior
global-HMM candidate: two states, 252 sessions, zero turnover band, monthly
rebalance, one-session lag, and 10 bps one-way costs.

## Development result: 2014-01-31 through 2019-12-31

| Strategy | Annualised return | Volatility | Sharpe | Maximum drawdown | 95% daily CVaR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group-aware HMM risk | 6.94% | 7.28% | 0.958 | -12.53% | **-1.08%** |
| Matched global-HMM risk | **7.06%** | 7.38% | **0.961** | **-12.47%** | -1.09% |

The paired 21-session moving-block bootstrap on 1,489 aligned daily net
returns gives an annualised mean-return difference of **-0.12%** and a Sharpe
difference of **-0.004** for group-aware minus global HMM. The 95% bootstrap
interval is [-0.00190%, 0.00103%] daily and the one-sided p-value for a
positive group-aware advantage is **0.732**.

The run produced 72 decisions with all five groups present. The PSD projection
was non-zero at 47 decisions (mean Frobenius adjustment `2.70e-05`), confirming
that the group block is active rather than a no-op. It nevertheless did not
meet the amendment's Sharpe/drawdown criterion.

## Decision

Archive the group-aware covariance construction as a negative ablation. Do not
retune the group taxonomy, HMM state count, lookback, or projection on this
development panel; those changes would create a new search family. The
candidate ledger is complete but explicitly non-selectable, so this result
cannot be used to launch a confirmation run or make an outperformance claim.
