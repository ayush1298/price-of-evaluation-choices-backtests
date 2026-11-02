# Track-B Robustness Matrix: cost, cap, and rebalance frequency

The experiment protocol requires the outer matrix to vary transaction cost,
portfolio cap, and rebalance frequency. Cost is covered by repricing stored
target streams, which needs no refit. Cap and frequency both change the
decision itself, so each is a full re-run.

These are reported robustness views. Nothing here may change which candidate
the confirmation cycle put forward, and no cell below is a selection.

```bash
python -B scripts/run_track_b_robustness_matrix.py
```

## Cap and frequency results (net Sharpe)

| Strategy | 10% cap, monthly (base) | 5% cap, monthly | 10% cap, quarterly |
| --- | ---: | ---: | ---: |
| Partial-correlation hub-avoid + mean-CVaR | **0.748** | 0.728 | 0.672 |
| PCC hub-avoid + mean-CVaR | 0.723 | 0.722 | 0.667 |
| Inverse volatility (252) | 0.708 | 0.708 | **0.674** |
| HRP (252) | 0.704 | 0.698 | 0.668 |
| HRP (756) | 0.695 | 0.699 | 0.665 |
| *HMM conditional risk (2/252)* | 0.697 | 0.696 | **fails** |
| *Soft-regime conditional HRP (756)* | 0.686 | 0.690 | **fails** |
| Equal weight | 0.678 | 0.678 | 0.646 |
| *Dynamic graph-temporal, seed 7* | 0.701 | 0.666 | see below |

Italicised rows are the frozen candidates.

The dynamic model's quarterly cell **cannot be placed in this column**, and the
reason is the most instructive thing the robustness matrix produced.

Tightening the cap from 10% to 5% moves every strategy by at most 0.020 Sharpe
and does not reorder the ladder's top or bottom. Quarterly rebalancing lowers
every surviving strategy by 0.03--0.08 Sharpe, and compresses the spread: the
ladder's range narrows from 0.070 (monthly) to 0.028 (quarterly), with
inverse-volatility overtaking the partial-correlation allocator. The
conclusion is unchanged in all completed cells — no candidate leads.

## The quarterly dynamic cell is not comparable, and why that matters

Run as specified, the quarterly dynamic model reports net Sharpe **1.185**
against a quarterly equal-weight figure of **0.646** in the column above — an
apparent edge of +0.54, which would be the largest positive result anywhere in
this study.

It is an artefact of the evaluation window. The model requires 24 completed
quarterly forward windows before its first decision, so its first signal is
2020-03-31 while the baselines start 2014-03-31:

| Run | First NAV date | Observations |
| --- | --- | ---: |
| Quarterly equal weight | 2014-04-01 | 3,097 |
| Quarterly partial-correlation/CVaR | 2014-04-01 | 3,097 |
| Quarterly dynamic, seed 7 | **2020-04-01** | **1,569** |

Re-running every baseline over the model's own window resolves it:

| Strategy (quarterly, first signal 2020-03-31) | Sharpe |
| --- | ---: |
| **Dynamic graph-temporal, seed 7** | **1.185** |
| HRP (252) | 1.168 |
| Inverse volatility | 1.162 |
| Equal weight | 1.140 |
| PCC hub-avoid + mean-CVaR | 1.139 |
| Partial-correlation hub-avoid + mean-CVaR | 1.135 |
| HRP (756) | 1.127 |

The +0.54 edge becomes **+0.017** over the strongest date-matched baseline. The
entire apparent advantage was the difference between a 2014-start and a
2020-start window, and nothing about the model.

What makes this dangerous is that no individual run is wrong. Each is correct,
manifest-backed, and reproducible. The error exists only in placing two of them
in the same column, which is invisible unless observation counts are printed —
so this study prints them. The effect is recorded as a row in
[`PROTOCOL_INFLATION_RESULTS.md`](PROTOCOL_INFLATION_RESULTS.md) at **+0.493
Sharpe**, the largest single inflation measured anywhere in this work.

## Two declared cells could not be run, and why

### A 2% cap is arithmetically impossible on this panel

The protocol declares caps of 2%, 5%, and 10%. A 2% cap across a 30-asset
universe permits at most `30 x 0.02 = 0.60` of the portfolio to be invested, so
no fully invested long-only allocation exists. The pipeline's own allocator
guard rejects the configuration rather than silently producing a partially
invested portfolio.

The minimum feasible cap on this panel is `1/30 = 3.34%`. The 2% cell is
therefore not a missing experiment; it is undefined for a universe of this
size. The declared value implicitly assumed a large equity cross-section.

### Both regime candidates fail under quarterly rebalancing

Both frozen regime-conditional candidates raise
`Conditional covariance needs at least two aligned historical returns` at their
first quarterly decision. The cause is structural rather than numerical: each
estimates its market state on the *rebalance* clock, so moving from monthly to
quarterly decisions cuts the state panel by two thirds while the warm-up
requirement of 36 state observations is unchanged. Thirty-six quarters is nine
years of history before a first state estimate, which the panel cannot supply
alongside the covariance lookback.

This is reported as a result, not repaired. Decoupling the regime clock from
the rebalance clock would be a reasonable design, but adopting it now would
re-specify a frozen candidate after its confirmation results were seen, which
the charter forbids.

The finding is worth stating plainly: the regime candidates are not merely
weaker than the baselines, they are **more fragile to a routine protocol
variation**. Every conventional allocator in the ladder absorbs a change of
rebalance frequency without modification; neither regime method does. A study
reporting only its preferred monthly configuration would never surface that.

## Transaction-cost robustness

Repriced from immutable target streams; no refit, graph rebuild, or
re-allocation. Mean one-way turnover per execution is shown for context.

| One-way cost | Equal weight | Partial-corr/CVaR | Soft-regime HRP | HMM risk | Dynamic seed 7 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bps | 0.680 | 0.760 | 0.692 | 0.703 | 0.735 |
| 5 bps | 0.679 | 0.754 | 0.689 | 0.700 | 0.718 |
| 10 bps | 0.678 | 0.748 | 0.686 | 0.697 | 0.701 |
| 25 bps | 0.676 | 0.731 | 0.678 | 0.687 | 0.651 |
| 50 bps | 0.673 | 0.703 | 0.663 | 0.672 | 0.568 |
| **Turnover / execution** | **2.0%** | 14.4% | 8.1% | 8.7% | **48.9%** |

The cost gradient tracks turnover almost exactly, which is the mechanism behind
the protocol-inflation table's cost row: the shortcut of reporting gross
returns is worth 0.007 Sharpe to equal weight and 0.167 to the dynamic model.

## Scope

Date-matched quarterly baselines are recorded under the `alt-etf-quarterly2020-`
prefix and exist solely to give the quarterly dynamic cell a legitimate
comparator. They are not an additional evaluation window.

The dynamic graph-temporal cap and frequency variants are run at **seed 7
only**. The full five-seed grid across both variants is several hours of
compute for a reported robustness view, and the primary window already
establishes that model's seed dispersion. The single-seed scope is disclosed
wherever those rows appear and must not be read as a seed-stable result.

All cells use the confirmation panel, the 2014-01-31 to 2026-07-24 window, one
session execution lag, and the shared allocator.
