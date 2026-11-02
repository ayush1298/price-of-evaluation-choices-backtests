# ICAIF Charter Amendment v2 — Group-Aware Conditional Risk

**Frozen on:** 30 July 2026
**Scope:** one Track-A development diagnostic only; no confirmation-market or
paper-result evaluation is authorised by this amendment.

## Reason for amendment

The v1 uncertainty-aware, market-wide HMM covariance model failed its
development gate. The failure may reflect noisy state-conditioned cross-asset
covariances across fundamentally different ETF exposures rather than a useful
common regime signal. The predeclared US ETF universe already contains stable
asset descriptions, so this amendment tests one structural alternative without
adding HMM states, lookbacks, neural architectures, or allocator choices.

All original public US ETF prices, prior HMM results, and baseline results were
already observed when this amendment was written. Thus the diagnostic cannot
be presented as an untouched result or used to open the alternative ETF
confirmation panel.

## Hypothesis and fixed implementation

The sole hypothesis is that a soft market regime should affect covariance
inside economically coherent asset groups more than cross-group covariance.

At each decision date the method will:

1. fit the same causal two-state expanding Gaussian HMM and use the same soft
   probability/uncertainty fallback as v1;
2. estimate the same state-conditional shrinkage covariance mixture;
3. retain the conditional mixture **only within** five ex-ante groups and use
   unconditional covariance between groups; and
4. pass that covariance to the unchanged 10%-capped HRP allocator, with the
   same monthly schedule, one-session execution lag, and 10 bps one-way cost.

The five groups are versioned in
[`us_etf_2010_v1_regime_groups_v1.csv`](../configs/universes/us_etf_2010_v1_regime_groups_v1.csv):
US core equity, US sector equity, international equity, fixed income, and
diversifiers. They derive from the predeclared instrument descriptions, not
from realised returns.

## Fixed diagnostic comparison

The only candidate configuration is recorded in
[`public_us_etf_group_aware_hmm_diagnostic_v1.json`](../configs/public_us_etf_2010_group_aware_hmm_risk_v1.json)
and its non-selectable candidate ledger. It is compared with the existing
matched global HMM candidate (two states, 252 sessions, zero turnover band) on
the fixed 2014-01-31 to 2019-12-31 development window.

Record net Sharpe, maximum drawdown, 95% daily CVaR, turnover, group count,
and a paired moving-block bootstrap of daily net returns. The diagnostic is
worth retaining only if it does not lower Sharpe or worsen tail risk/drawdown
and the paired comparison supplies credible directional evidence. Otherwise it
is archived as another failed ablation.

No other parameter values, grouping variants, or confirmation results may be
tested under this amendment. Any future confirmation protocol needs a fresh
amendment, a complete selectable ledger, and the data gates from charter v1.

The completed diagnostic and its negative decision are recorded in
[`GROUP_AWARE_HMM_RISK_RESULTS.md`](GROUP_AWARE_HMM_RISK_RESULTS.md).
