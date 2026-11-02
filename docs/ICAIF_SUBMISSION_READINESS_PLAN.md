# ICAIF Submission Readiness Plan

**Status on 28 July 2026: not ready for a paper-result table.**  The
repository now has a strong causal experiment framework and useful negative
evidence, but no proposed regime-aware, graph, or deep method has cleared its
held-out statistical and economic-performance gates.  The largest remaining
risk is data validity, followed by the absence of a positive, robust research
result.  More architecture tuning before resolving those two points would not
make the eventual submission stronger.

This document closes the current evidence-collection task and defines the
work required before an ICAIF submission decision.  It deliberately separates
an exploratory result from a publishable one.

## 1. What is already in the project

### Core research infrastructure to retain

The new `src/research/` pipeline is the foundation to keep.  It has a common
walk-forward backtest, constrained allocator, graph construction, causal
regime models, model-training guardrails, evaluation metrics/inference, and a
manifest for reproducibility.  All models are intended to use the same
eligible universe, trading calendar, execution rule, portfolio caps, and
transaction-cost assumptions.

The following controls should remain non-negotiable:

1. **Causal timeline.** At a decision date the pipeline may see only data then
   available; fitting scalers, regime models, graphs, hyperparameters, and
   normalisers on future observations is prohibited.
2. **One shared allocator and backtester.** A learned score is sent through
   the same constrained mean-CVaR allocation machinery as a conventional
   method.  A model may not appear better merely because it received looser
   constraints or a more favourable backtest.
3. **Manifest-backed runs.** Configurations, input hashes, code revision,
   weights, executions, holdings, daily NAV, diagnostics, and seeds are
   recorded for a canonical result.
4. **Point-in-time eligibility and identity mapping.** Membership, ticker
   changes, delistings, calendar, adjustment convention, and source hashes
   are part of the data contract rather than hidden preprocessing details.
5. **No automatic promotion.** A method must improve a matched baseline on an
   outer test, after costs and inference, before it can be called the proposed
   method.
6. **Explicit paper gate.** `paper_eligible` is false by default.  The
   existing protocol requires verified universe and price adjustment,
   precommitted evaluation, independent confirmation, and completed
   robustness checks before a table can be exported.

The legacy `src/mtp1`, `src/mtp2`, dashboard, and `results/legacy` material
should be retained as historical traceability and illustrative visualisation,
but not removed or presented as canonical evidence.  In particular, their
post-hoc/small-sample outputs must not enter a manuscript performance table.

### Current data-integrity work: completed versus open

The bounded data-audit task is complete.  It established that the official NSE
bhavcopy archive can be fetched for every sampled month from January 2008 to
August 2025, including the legacy/UDiFF format break.  It also produced a
34,425-record corporate-action review ledger, a strict candidate identity
crosswalk, explicit action-application safeguards, and daily observations
around several historical code transitions (for example `HEROHONDA` to
`HEROMOTOCO`, `VSNL` to `TATACOMM`, and `ZOMATO` to `ETERNAL`).  The most
recent unit-test run passed 66 tests.

This is evidence collection, *not* completion of the Indian equity panel.
The following are intentionally still blocked:

- candidate mappings must be upgraded to verified mappings using dated,
  primary-source evidence;
- each split/bonus must be reviewed, while rights issues, mergers, demergers,
  dividends, and delistings require their own treatment rather than a blind
  multiplicative adjustment;
- the reconciled daily series must be checked against a documented
  split/dividend-adjusted total-return source;
- point-in-time Nifty membership, including exits/delistings, must be
  independently verified.

No corporate action has been silently applied and no candidate crosswalk is
executable by default.  That conservative behaviour is correct and should be
kept.

## 2. What we tried and what the evidence says

All figures below are internal exploratory diagnostics, not paper claims.
The exact run definitions and caveats are in the linked result documents.

| Workstream | What was tried | Result | Interpretation |
| --- | --- | --- | --- |
| Current S&P 500 Kaggle panel | 88 sector-balanced current survivors, monthly rebalance, 10 bps cost, 5% cap; equal weight, inverse volatility, HRP, and PCC/CVaR | The development winner PCC/CVaR was only marginally above HRP (Sharpe 1.087 vs. 1.082; paired p=0.470). On frozen 2020–25 data, HRP Sharpe was 0.435 versus 0.612 for equal weight. | Useful engineering diagnostic; invalid as main evidence because it is a current-constituent survivor universe. |
| Community Nifty membership plus Yahoo prices | 100 community membership tickers; 83 price series; equal weight, inverse volatility, HRP, PCC/CVaR | Development PCC/CVaR Sharpe 1.259 versus HRP 1.047, but p=0.051 and deflated-Sharpe probability 0.430. Frozen 2020–Aug 2025: equal weight 1.137, HRP 1.261, PCC/CVaR 1.183. | The apparent development gain did not replicate.  The sample also had only 40–48 tradable sources per rebalance, so it is not an index replication. |
| Regime routing | Causal expanding K-Means, HMM hard routing, HMM probability mixture, soft risk regimes | None delivered a stable conditional advantage over regime-free, equal-weight, or PCC/CVaR comparators.  For India ETFs, equal weight (1.528) and PCC/CVaR (1.633) exceeded the HMM mixture (1.486). | The current regime feature is not yet a source of portfolio value. |
| Graph attention / DeepCNL-style comparator | Causal graph-attention allocation against matched similarity graph | US development Sharpe: similarity 0.821, attention 0.807; p=0.474. | No evidence that the added graph-attention complexity helps. |
| Information-coefficient objective | Train a dynamic model on an IC-oriented loss | US change was tiny (0.775 to 0.788; p=0.298); India was worse (1.297 to 1.230; p=0.998 for superiority). | Do not advance this as a solution. |
| Gated neural experts | Regime-free, direct-concatenation, and gated-expert dynamic models | US 2016–19 development Sharpes: 1.021, 0.843, and 0.834 respectively. | Gating has so far made the model worse. |
| Hierarchical risk parity | HRP across 30/50 US ETFs and Indian ETF panels | Development differences were small; outer tests were below equal weight (for example 50 ETF: 0.760 HRP vs. 0.822 equal weight). | A necessary baseline, not a validated winner. |
| Legacy RNN/LSTM/GRU and corrected Transformer smoke tests | Repaired warm-up/master-universe issues and ran limited neural diagnostics | The usable common universe became too short/small for a meaningful comparison; no model was promoted. | The legacy neural models are not a credible paper contribution in their current form. |

### What worked

- The refactored pipeline exposes leakage, data availability, universe changes,
  and transaction-cost differences rather than concealing them.
- Strong simple comparators often beat complex methods.  This is a useful
  empirical result because it narrows the research question and prevents an
  unsupported deep-learning claim.
- Development/outer separation and paired inference caught non-replicating
  gains before they could become a manuscript conclusion.
- Official NSE archive and corporate-action collection is now reproducible,
  reviewable, and intentionally fails closed.

### Performed below expectation / worst failure modes

1. No model has a statistically credible, cost-aware outer-test advantage.
   The poorest direction is treating a complex regime/graph/neural allocation
   as a replacement for a robust simple allocator.
2. The Kaggle S&P dataset has survivor bias for this purpose; the community
   Nifty file and Yahoo coverage are insufficient for an historical index
   backtest.  Neither can support the central performance claim.
3. The public ETF panels are valuable reproducible prototypes but are not yet
   documented as a point-in-time, corporate-action-complete paper universe.
4. The old RNN/LSTM/GRU outputs lack a matched causal experiment and should
   not be rescued through selective retuning.
5. A single favourable development Sharpe, regime label, or chart is not
   evidence.  The project already has several examples of that gain vanishing
   on the frozen period.

## 3. Recommended submission contribution

Do not frame the paper as “deep co-investment networks outperform portfolio
benchmarks.”  The present evidence does not support it.  The testable, more
credible contribution is:

> **Can uncertainty-aware, causally estimated market regimes improve
> risk/covariance forecasts and constrained portfolio decisions beyond strong
> non-regime baselines, after transaction costs and point-in-time data
> controls?**

The proposed method should be compact and structural:

1. infer a **soft causal latent state** from market-wide features available at
   each decision date (returns, realised volatility, drawdown, cross-sectional
   dispersion/correlation, and—where independently sourced—VIX/India VIX);
2. estimate **state-conditional, shrinkage covariance/risk forecasts** and
   blend them by state probabilities, including an uncertainty/no-trade or
   conservative-shrinkage rule when state probabilities are diffuse;
3. use the existing constrained long-only mean-CVaR or minimum-risk allocator,
   with turnover penalty, concentration caps, and common execution;
4. treat a dynamic graph as a *forecast feature/ablation*, not as automatic
   evidence that a graph network should generate portfolio weights directly.

This has two advantages: it makes the source of value observable (forecast
calibration, tail-risk control, and turnover), and it has a serious simple
baseline.  It also uses the current causal routing and allocator rather than
discarding the completed engineering work.

A temporal neural model may be one predeclared comparator, not the centre of
the paper.  Use a compact TCN/Transformer or gated recurrent unit with the
same inputs and allocator, fixed by the validation protocol.  Do not run a
large model zoo of RNN/LSTM/GRU/Transformer variants: that increases selection
bias and has already failed to give a signal.  If the compact neural model
does not pass validation and its outer test, report it as an ablation and keep
the statistical method as the proposed model.

Recent work already represented in the protocol supports this direction:
graph-attention portfolio work motivates a matched graph comparator;
multivariate HMM work motivates sector heterogeneity; implementable-efficient-
frontier research motivates common cost-aware implementation; and recent
regime-aware work emphasises causal state identity and probability allocation.
Before drafting, perform a fresh literature search for the submission year and
write a one-page novelty matrix: paper, data, regime construction, allocator,
evaluation, and the precise difference from this work.  Add only methods that
change a preregistered hypothesis—not a method merely because it is recent.

## 4. Data decision and OpenBB assessment

### Canonical-data requirement

For every paper market, create a versioned data contract containing:

- a point-in-time constituent/membership history, including additions,
  removals, delistings, ticker/ISIN changes, and the as-of availability date;
- daily OHLCV and a documented adjustment convention.  A total-return close
  or an independently auditable split/dividend adjustment is required for a
  return study;
- raw-file and processed-panel hashes, source/licence, retrieval date,
  calendar/time-zone, currency/FX rule, and treatment of non-trading/missing
  observations;
- liquidity/volume information where capacity is claimed; otherwise make no
  capacity claim;
- an eligibility mask frozen from the first model signal date, not a list of
  today's surviving tickers.

The first realistic target is **one fully verified market** before attempting
four.  A second independent market is then confirmation, not an opportunity
to tune.  If a licensed historical stock database or a university terminal
becomes available, use it for the canonical large-cap equity market.  If it
does not, a public-data paper remains possible only with a deliberately
reproducible fixed instrument universe (for example, liquid ETFs selected by
an ex-ante vintage rule) and a modest claim; it is weaker than a
point-in-time-stock-universe study and should not claim index replication.

The official NSE archive work should continue as an India extension only once
the member history and corporate-action reconciliation can meet this same
contract.  The current Kaggle files remain exploratory robustness/smoke-test
inputs, never headline evidence.

### Where OpenBB fits

[OpenBB](https://github.com/OpenBB-finance/OpenBB) is an open-source provider
integration layer, not a historical-market-data repository.  Its historical
equity-price endpoint can use providers including yfinance, Alpha Vantage,
Tiingo, FMP, Cboe, TMX, and Tradier; provider coverage, adjustment quality,
licensing, and API credentials remain the responsibility of the user.  Its
documentation also makes clear that it does not host or guarantee the data.
The relevant price endpoint is documented at
<https://docs.openbb.co/odp/python/reference/equity/price/historical> and the
provider extension model at
<https://docs.openbb.co/odp/python/extensions/providers>.

Therefore OpenBB **does not solve** the critical gaps: historical Nifty/S&P
membership, delistings, dated ticker/ISIN continuity, NSE corporate actions,
or a validated total-return history.  It mostly exposes yfinance without a
key, which the project has already used.  It should not be cited as the source
of a canonical observation; the actual provider and raw retrieval must be
recorded.

It is still useful in three limited ways:

1. add an *optional isolated adapter* for public ETF/macro prices so the
   pipeline can switch between yfinance and a licensed provider without
   changing its data schema;
2. use a second provider, when credentials exist, to cross-check a small
   sample of prices/actions and current metadata/ISINs;
3. make a reproducibility example easier for readers to run on a public
   non-paper panel.

Do **not** integrate OpenBB into the canonical pipeline now.  First freeze the
data contract and choose the actual provider.  If it is later distributed as
part of this repository, review OpenBB's AGPL-3.0 licence and the project
licence with the supervisor before merging code.  An adapter must retain the
real provider, parameters, retrieval timestamp, raw response hash, and
adjustment setting in the manifest.

## 5. Work plan and acceptance gates

The stages are intentionally sequential where later work could otherwise
overfit or hide a data error.  A failed gate is a result: record it, do not
quietly replace the protocol.

| Order | Work and deliverable | Acceptance gate |
| --- | --- | --- |
| 0 | Freeze the research question above, primary metric (net Sharpe plus tail-risk/turnover), markets, costs, and a short candidate list in a versioned experiment charter. | Supervisor signs off on the claim and data scope; no performance-based changes afterwards. |
| 1 | Select the canonical first market/provider.  Write `DATA_CONTRACT_v1` with licence, history, membership, pricing, adjustments, availability timing, and exclusions. | A reviewer can recreate one monthly rebalance's eligible assets and adjusted returns from raw files. |
| 2 | Finish the India identity/action work or document why a different primary market is chosen.  Upgrade mappings only with dated evidence; add action-review decisions and independent spot checks. | No unreviewed candidate mapping/action is applied; all unresolved events are excluded or explicitly handled. |
| 3 | Build a validated, point-in-time daily panel and eligibility masks.  Add data-quality report: coverage, missingness, delistings, membership counts, duplicate ISINs, return outliers, and price-source reconciliation. | Data validation passes and a reproducible raw-to-panel manifest is saved. |
| 4 | Precommit a rolling-origin design: recommended five-year expanding train, one-year validation, one-year outer test, advanced yearly; five seeds for stochastic models. | An outer-test configuration cannot be changed using its own results. |
| 5 | Re-establish the baseline ladder on the new panel: equal weight, inverse volatility, minimum variance/risk parity/HRP, and long-only mean-CVaR with the same costs/caps. | Baselines reconcile to their saved trades/NAV; all outer years have daily return series. |
| 6 | Add the proposed uncertainty-aware regime-conditional covariance/risk estimator.  Predefine no-regime, hard/soft HMM, and non-graph/graph ablations. | It improves validation forecast and/or allocation criteria against the predeclared matched baseline without a turnover artefact. |
| 7 | Run one compact temporal neural comparator and the existing dynamic-graph comparator under identical data/allocator conditions.  Archive rather than tune failures. | Only models selected on validation enter the untouched outer test. |
| 8 | Evaluate outer tests across costs (0/5/10/25/50 bps), caps (2/5/10%), monthly/quarterly rebalancing, lookbacks (63/126/252), seeds, and a second market. | Report net/gross return, volatility, Sharpe, Sortino, Calmar, drawdown, CVaR, turnover, cost, holdings, regime time, and failure cases. |
| 9 | Perform paired block-bootstrap inference, deflated-Sharpe/probability-of-backtest-overfitting diagnostics, and White Reality Check or Hansen SPA across all tried candidates.  Add factor exposures if factor data are properly aligned. | The selected method has credible economic magnitude and survives multiplicity-aware inference; otherwise change the paper to a negative/diagnostic study or do not make an outperformance claim. |
| 10 | Independent replication: rerun from an empty results directory or by a collaborator on a second machine/provider snapshot.  Export only manifest-backed paper-eligible rows. | All five existing paper-evidence flags are true for every manuscript result. |
| 11 | Write the paper: problem/data contract, causal method, baselines, split protocol, complete results/ablations, limitations, reproducibility appendix, and data/code availability statement. | No table/chart comes from `results/legacy`, a static survivor universe, post-hoc regime labels, or an unreviewed exploratory panel. |

## 6. Model-selection rules for the next implementation cycle

The project may keep and repair the current pipeline rather than rewrite it.
The decision is **replace selected modules, not the experimental spine**:

| Stage | Keep | Change / add |
| --- | --- | --- |
| Data ingress | Schema, provenance, strict crosswalk/action guards, archive collectors | Canonical provider adapter, verified membership/total-return source, full quality report |
| Features | Causal returns/volatility/correlation graph snapshots | Cross-sectional dispersion, liquidity where licensed, state uncertainty features, sector-aware inputs |
| Regime | Expanding K-Means/HMM as transparent baselines | Soft probabilistic state model with state-identity tracking; optional online change-point/HSMM comparator |
| Risk forecast | Shrinkage/PCC and covariance controls | State-conditional shrinkage covariance blended by probabilities; calibration and tail-forecast diagnostics |
| Allocation | Shared constrained mean-CVaR engine, turnover/cap controls | No-trade band and robust/uncertainty penalty as predeclared allocator ablations |
| Deep models | Existing Transformer/graph attention only as fair matched comparators | At most one compact causal TCN/GRU/Transformer comparator; no direct unregularised weight prediction |
| Evaluation | Walk-forward engine, manifests, costs, bootstrap comparison, deflated Sharpe | Complete SPA/Reality Check, robustness matrix, independent market and replication release |

## 7. Practical no-go / go decision

Proceed to a full ICAIF manuscript only if all of the following are true:

1. at least one market has verified point-in-time membership and adjusted
   price data;
2. the selected method is chosen without examining its outer-test years;
3. it beats the matched non-regime baseline on a meaningful net measure and
   does not merely transfer risk to turnover/drawdown/tail loss;
4. the gain survives the predefined robustness and multiple-testing checks;
5. it independently replicates in another market or defensibly scoped sample;
6. all data, model, and evaluation limitations are disclosed.

If these gates are not met, the best next product is a transparent research
report or an ICAIF-style reproducibility/negative-evidence submission only if
the novelty is the audited benchmark and diagnostic methodology.  It should
not claim superior regime-aware portfolio performance.

## 8. Immediate next actions

1. Ask the supervisor/data librarian for a historical point-in-time equity
   source (membership plus adjusted OHLCV/total returns) and clarify the
   licence for research reproduction.
2. Decide whether the primary paper market is a licensed historical-equity
   universe or an explicitly limited public fixed-instrument study.
3. Finalise the NSE crosswalk/action review only if India remains in scope.
4. Create the experiment charter and data contract before running another
   model.
5. Once the new panel passes validation, rerun the baseline ladder first;
   promote no neural/regime candidate until that comparison is complete.

The detailed experiment conditions are maintained in
[`ICAIF_EXPERIMENT_PROTOCOL.md`](ICAIF_EXPERIMENT_PROTOCOL.md); exploratory
results and their limitations remain documented in the individual results
files rather than being relabelled as paper evidence.
