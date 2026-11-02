# ICAIF Experiment Charter v1

**Frozen on:** 28 July 2026  
**Status:** precommitted next-cycle design; no result generated under this
charter is a paper result until the data and confirmation gates are met.

## 1. Research question and permitted claim

The next research cycle asks:

> Does a causally inferred *soft* market regime improve net portfolio
> risk-adjusted performance and tail-risk control beyond a matched
> non-regime allocation, once transaction costs, turnover, and point-in-time
> eligibility are held constant?

The permitted claim is about a **fixed, pre-declared ETF allocation universe**
until a separately validated historical-equity data contract is supplied.  It
is not a claim about historical S&P 500, Nifty 50, or stock-index membership.
No model may be called superior because of a development-period Sharpe, a
post-hoc regime label, an unpaired comparison, or a dashboard chart.

## 2. Tracks and data boundary

### Track A — public development and reproducibility

The only active no-cost dataset is the 30-instrument US ETF universe in
`configs/universes/us_etf_2010_v1.csv`, with public yfinance OHLCV and a
pre-declared fixed-membership rule.  Its exact data contract is
`configs/contracts/public_us_etf_2010_v1.json`.

All observations and prior results from this panel are considered seen
development evidence.  They can improve code, check data quality, select a
small hypothesis set, and reproduce ablations.  They cannot constitute an
untouched final paper test.

### Track B — paper confirmation

The paper confirmation must use either:

1. a previously unexamined, independently documented fixed-ETF market/panel;
   or
2. a licensed or independently audited point-in-time historical-equity panel
   with membership, delistings/identity continuity, and adjusted total-return
   prices.

The panel, contract hash, first signal date, and all candidate configurations
must be frozen before Track-B results are viewed.  India remains an extension
candidate; it is ineligible until the NSE membership, crosswalk, and
corporate-action gates pass.

The first Track-B candidate is the distinct, predeclared ETF universe in
`configs/universes/us_etf_alternative_2010_v1.csv`, governed by
`configs/contracts/public_us_etf_alternative_2010_v1.json`.  It shares no
traded ticker with Track A.  Its instrument list was written before source
retrieval and the returned panel has passed its schema/coverage contract; no
return or model evaluation may be inspected until the selected method is
frozen on Track A.

## 3. Candidate method and comparisons

### Proposed statistical method

The proposed method is an **uncertainty-aware soft-regime conditional risk
forecast**:

1. calculate only causal market features: trailing market return, realised
   volatility, drawdown, cross-sectional dispersion/correlation, and optional
   independently available volatility-index data;
2. estimate soft state probabilities with an expanding Gaussian HMM; state
   count is selected only in the current training/validation span;
3. estimate shrinkage covariance/risk forecasts per state and blend them by
   those probabilities;
4. use a conservative/no-trade rule when state probabilities are diffuse;
5. send the risk/return inputs to the same long-only, capped, turnover-aware
   constrained allocator as every comparator.

The model must show its mechanism through regime probabilities, covariance
calibration, realised CVaR, concentration, and executed turnover.  A direct
neural portfolio-weight output is not permitted.

### Predeclared controls

All use the same universe, observed price dates, one-session execution lag,
monthly rebalancing, 10 bps one-way costs, and 10% maximum weight in the
initial study:

1. equal weight;
2. inverse volatility;
3. minimum-risk / Hierarchical Risk Parity;
4. no-regime shrinkage-covariance allocation using the identical allocator;
5. PCC and partial-correlation mean-CVaR controls; and
6. hard HMM routing, to separate soft uncertainty blending from merely adding
   an HMM.

Only one neural comparator is allowed: the existing causal dynamic
graph-temporal model with exactly the same data, allocator, cost, and
selection timeline.  It is an ablation; it is not the proposed method unless
it independently clears every selection gate.

## 4. Selection and outer-test protocol

For Track B, use five years expanding training, one year validation, and one
year outer test, advanced annually where the panel has sufficient history.
Candidate values are limited to:

- HMM states: 2, 3, 4;
- covariance lookback: 63, 126, 252 sessions;
- CVaR aversion/turnover penalty: values listed in the frozen run config;
- no-trade band: 0%, 10%, 25%; and
- graph edge fraction: 5%, 10% only for graph ablations.

Select a configuration exclusively by validation performance using the
predeclared criterion below.  Freeze it for the paired outer test.  Stochastic
models use seeds 7, 17, 29, 43, and 59; report median, IQR, and worst seed.
Every attempted configuration, including failures, belongs in the candidate
ledger.

The primary decision statistic is a paired, moving-block-bootstrap test of
the **net daily return differential** between the proposed soft-regime method
and the selected matched no-regime control.  The target is a positive
differential at one-sided 5% significance, with annualised net Sharpe no
lower than the control.  It must additionally not worsen 95% realised daily
CVaR or maximum drawdown by more than the tolerance declared in the final
Track-B config.  This prevents a cosmetic Sharpe gain created by tail risk.

Apply White's Reality Check or Hansen's SPA to the complete validation
candidate set, and report deflated Sharpe/probability-of-backtest-overfitting.
The full outer-test matrix must also vary costs (0/5/10/25/50 bps), cap
(2%/5%/10%), and rebalance frequency (monthly/quarterly).

## 5. Evidence already seen and excluded from selection

The Kaggle-current-S&P, community-Nifty/Yahoo, public-US-ETF, public-India-ETF,
and legacy panels have been inspected.  Their graph, HMM, gated-expert,
information-coefficient, HRP, RNN/LSTM/GRU, Transformer, and graph-attention
results are development/diagnostic evidence only.  No prior apparent winner
is grandfathered into Track B.  The run manifests and the results notes are
the selection ledger.

## 6. Data and reproducibility requirements

Before any Track-B run:

1. validate a versioned data contract with
   `scripts/validate_data_contract.py`;
2. place the generated contract-validation report, raw source sidecars, and
   contract JSON in the run manifest inputs;
3. run `scripts/validate_market_dataset.py` as a second market-coverage
   check;
4. save all target weights, executions, holdings, daily NAV, model/regime
   diagnostics, configs, code revision, and random seeds; and
5. set every run to `paper_eligible: false` until all five mechanical release
   evidence flags in `ICAIF_EXPERIMENT_PROTOCOL.md` are demonstrably true.

## 7. Stop/go rules

Stop the outperformance storyline if the method fails the outer paired test,
fails multiplicity-aware inference, is materially more costly/unstable, or
does not confirm on the second market.  Preserve the failure and report it as
an ablation.  Proceed to paper writing only when the full protocol and all
data-contract release conditions pass.

This charter is deliberately more restrictive than the prior exploratory
workflow.  Any amendment requires a new version, a reason, and a statement of
which results had already been observed when it was made.
