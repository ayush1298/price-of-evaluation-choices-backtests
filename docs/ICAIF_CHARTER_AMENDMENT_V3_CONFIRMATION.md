# ICAIF Charter Amendment v3 — Track-B Confirmation Protocol

**Frozen on:** 6 August 2026, before any return, metric, or model output from
the alternative ETF panel has been computed or viewed.
**Scope:** the single confirmation cycle on
`configs/universes/us_etf_alternative_2010_v1.csv`.

## 1. Why this amendment exists

Charter v1 §2 reserved a Track-B confirmation panel and required that the
method be frozen on Track A before the panel is opened. Charter v1 assumed the
frozen method would be a *winner*. It is not: every proposed regime-aware,
graph, and neural candidate failed its Track-A development gate, and the
multiplicity-corrected inference over the HMM search family gives a family-wise
p of 0.680.

This amendment resolves the resulting ambiguity. Track B is opened not to
confirm a winner but to test a **pre-registered prediction of failure**. That
prediction is falsifiable, which is what makes reporting its outcome
meaningful.

## 2. Pre-registered hypothesis

> **H1 (predicted to fail).** On the alternative ETF panel, at least one
> Track-A-frozen regime-aware, graph, or neural candidate exceeds the best
> pre-declared simple baseline on net annualised Sharpe, with a one-sided
> paired moving-block bootstrap p < 0.05 on net daily return differentials, and
> without worsening 95% daily CVaR or maximum drawdown.
>
> **H0 (predicted outcome).** No such candidate exists.

We predict H0. If H1 holds for any candidate, that is a genuine surprise and
must be reported as such — including the observation that the same candidate
failed on Track A, which would itself be evidence about the noise level of this
class of study rather than a validated method.

Recording the direction of the prediction before opening the panel is the point
of this document. A result that merely *confirms* an expectation is only
informative if the expectation was written down first.

## 3. Panel and its independence

The panel is 30 US-listed ETFs declared before source retrieval, governed by
`configs/contracts/public_us_etf_alternative_2010_v1.json`. It shares **zero
traded tickers** with the 30-instrument Track-A development universe
(verified: set intersection is empty). Its composition is deliberately
different in character — style, industry, and single-country exposures rather
than the broad asset-class mix of Track A.

As of this freeze, no run directory under `results/canonical/` references this
panel, and no return, metric, graph, regime label, or model output from it has
been computed or inspected.

## 4. Exact permitted run set

No run outside this list may be executed on the panel. Every run uses the
shared allocator, one-session execution lag, monthly rebalance, 10 bps one-way
cost, and 10% cap.

**Baselines (six):** equal weight; inverse volatility (252); HRP (252);
HRP (756); PCC hub-avoid + mean-CVaR; shrinkage partial-correlation hub-avoid +
mean-CVaR.

**Frozen candidates (three families):**

1. soft-regime conditional HRP, 756-session history — the Track-A
   configuration in `public_india_etf_2014_soft_regime_conditional_hrp_v1.json`
   applied unchanged to this market's config;
2. uncertainty-aware HMM conditional risk at **two states, 252 sessions, zero
   turnover band** — the Track-A development maximum, frozen and not re-searched;
3. causal dynamic graph-temporal model, five epochs, seeds 7/17/29/43/59 —
   reported as median, IQR, and worst seed, never best seed.

No other state count, lookback, band, epoch budget, edge fraction, architecture,
or seed may be run on this panel. A configuration that fails is archived, not
retuned. There is no second attempt.

## 5. Evaluation

**Primary window.** First signal 2014-01-31 through the end of the validated
panel, matching the Track-A development start so every method receives the same
warm-up as its Track-A counterpart. The entire window is held out: nothing on
this panel has been examined, so no development/outer split is required within
it.

**Secondary window.** The 2020-01-31 signal boundary, reported for
comparability with the Track-A chronological results.

**Primary statistic.** Paired 21-session moving-block bootstrap, 5,000
replications, seed 7, on net daily return differentials against equal weight,
computed only over dates both series realise. Each candidate is additionally
compared against the best-performing baseline on this panel.

**Multiplicity.** A White Reality Check over the complete confirmation
candidate set against the equal-weight benchmark, using the ledger
`configs/ledgers/public_us_etf_alternative_confirmation_v1.json`. Every run in
that ledger must be present; an incomplete family blocks the inference rather
than shrinking it.

**Reporting.** Net and gross annualised return, volatility, Sharpe, Sortino,
Calmar, maximum drawdown, 95% daily CVaR, turnover, realised cost, and average
holdings for every run, including failures.

## 6. What this cannot establish

The panel remains a fixed US-listed ETF allocation universe. It is not a
point-in-time equity-index constituent study, and confirmation here does not
discharge the membership, delisting, identity-continuity, or total-return
requirements in charter v1 §4. Both panels draw on the same provider
(`yfinance`) and overlap in time, so this is an independent *instrument* set,
not an independent data source or independent market regime history.

Consequently no result from this cycle may be described as a general
outperformance claim, and `paper_eligible` stays false for every run until the
release conditions in the data contract are met.

## 7. Stop rule

This is the only confirmation cycle authorised for this panel. Whatever the
outcome, the panel is spent: it becomes seen evidence the moment the first
metric is read, and no later amendment may reopen it as a fresh test. A further
confirmation requires a genuinely new panel, a new contract, and a new
amendment.
