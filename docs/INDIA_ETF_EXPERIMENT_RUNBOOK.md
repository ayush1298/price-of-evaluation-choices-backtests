# India fixed-ETF public experiment runbook

This is a second, separate public-data market study. It evaluates a fixed
pre-declared set of 12 India-listed ETFs in INR; it is not an NIFTY constituent
study and must never be pooled with USD results without an explicit FX policy.

## Data contract

`configs/universes/india_etf_2014_v1.csv` declares the ETF universe before
download. Yahoo Finance supplied adjusted OHLCV from 2014-05-05 through
2026-07-24. Because several ETF rows were intermittently missing, the
synchronised-panel gate retained only dates on which every declared ETF had an
observed OHLCV row; no value is filled, carried forward, or imputed.

The resulting local data package has 34,392 OHLCV rows, 2,866 shared trading
dates, 12 assets, and 147 strict monthly membership snapshots. It passes the
point-in-time validation gate with a minimum of 12 eligible assets and zero
membership staleness.

## Experimental rule

Use the India configurations and identical monthly timing, 10 bps one-way
costs, 10% cap, and five seeds as the US public-ETF experiment. Report INR
metrics separately. Start the diagnostic chronological comparison at
`2020-01-31`, but do not describe it as a fully untouched test until the
configuration is frozen independently of this market.

The public source remains unsuitable for historical NIFTY-stock membership or
delisting claims. It is an auditable ETF-allocation robustness market only.

## First chronological evidence (diagnostic)

The following manifest-backed runs begin at the 2020-01-31 signal boundary and
use 10 bps one-way costs. They span 1,577 matched daily INR observations.

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 18.93% | 14.17% | 1.295 | -23.77% |
| Inverse volatility | 18.57% | 13.91% | 1.294 | -22.99% |
| PCC hub-avoid + mean-CVaR | 19.58% | 14.41% | 1.313 | -20.40% |
| Partial-correlation hub-avoid + mean-CVaR | 18.64% | 13.95% | 1.295 | -27.14% |
| HMM mixture, five-seed mean | 18.58% | 14.88% | 1.220 ± 0.003 | -26.64% |
| Full graph-temporal, five-seed mean | 19.20% | 14.36% | 1.297 ± 0.056 | -20.91% |

For seed 7, the full graph-temporal model has Sharpe 1.259, graph-only 1.240,
and temporal-only 1.235. Paired moving-block bootstrap comparisons do not
support a positive mean-return advantage for the full model against either
ablation (`p=0.167` and `p=0.283`) or against PCC/CVaR (`p=0.650`). Two full
seeds exceed PCC/CVaR, but the five-seed mean does not. The model therefore
does not yet have a robust cross-market outperformance claim.

Mean monthly one-way turnover for the full model is 12–13% across seeds,
substantially below the US dynamic model. The India gap is consequently not
explained primarily by cost sensitivity; a better forecasting/regime mechanism
is required. The HMM's repeated numerical non-convergence warnings also mean
it should remain a transparent baseline, not a result to promote.

The later soft-regime-conditioned dynamic-model diagnostic uses a shorter
common realised period and is reported separately in
[`REGIME_CONDITIONED_DYNAMIC_RESULTS.md`](REGIME_CONDITIONED_DYNAMIC_RESULTS.md).
It does not improve the regime-free model or the simple India baselines.

The US-developed ranking-aligned likelihood/IC objective is evaluated on this
same outer India period in
[`INFORMATION_COEFFICIENT_MODEL_RESULTS.md`](INFORMATION_COEFFICIENT_MODEL_RESULTS.md).
It also fails to improve the regime-free dynamic model.

The matching public India VIX covariate package and the exact-date causal
merge contract are recorded in
[`EXTERNAL_REGIME_COVARIATES.md`](EXTERNAL_REGIME_COVARIATES.md).

The conventional Hierarchical Risk Parity allocation robustness result is
recorded in [`HIERARCHICAL_RISK_PARITY_RESULTS.md`](HIERARCHICAL_RISK_PARITY_RESULTS.md).

The soft-regime conditional covariance/HRP development diagnostic and its
unconditional-risk control are recorded in
[`SOFT_REGIME_CONDITIONAL_RISK_RESULTS.md`](SOFT_REGIME_CONDITIONAL_RISK_RESULTS.md).
