# Track-B Confirmation Results

## Status

**H0 confirmed. No frozen candidate beat the baseline ladder.** The panel is
now spent and cannot be reused as a held-out test.

This is the single confirmation cycle authorised by
[`ICAIF_CHARTER_AMENDMENT_V3_CONFIRMATION.md`](ICAIF_CHARTER_AMENDMENT_V3_CONFIRMATION.md),
executed on the alternative ETF panel on 6 August 2026. The amendment, the
permitted run set, and the prediction of failure were committed to version
control before the first metric on this panel was computed.

## What was pre-registered

The amendment recorded H1 — that at least one Track-A-frozen candidate would
beat the best pre-declared simple baseline with a one-sided paired
moving-block p < 0.05 and without worsening tail risk — and predicted H0, that
no such candidate exists. Recording the direction of the prediction beforehand
is what makes this outcome reportable rather than merely unsurprising.

The panel is 30 US-listed ETFs sharing **zero traded tickers** with the
development universe, governed by
`configs/contracts/public_us_etf_alternative_2010_v1.json`.

## Results ladder

First signal 2014-01-31, open-ended to the end of the validated panel. Net of
10 bps one-way costs, 10% cap, one-session execution lag, monthly rebalance.
Non-neural runs realise 3,137 daily observations; the dynamic model realises
3,120 because it requires a completed forward label.

| Strategy | Ann. return | Volatility | Sharpe | Max drawdown | 95% daily CVaR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Partial-correlation hub-avoid + mean-CVaR | 10.79% | 15.27% | **0.748** | -34.29% | **-2.29%** |
| PCC hub-avoid + mean-CVaR | 10.41% | 15.34% | 0.723 | -35.98% | -2.32% |
| Inverse volatility (252) | 11.27% | 17.19% | 0.708 | -38.33% | -2.57% |
| HRP (252) | 10.79% | 16.51% | 0.704 | -37.69% | -2.48% |
| *HMM conditional risk (2 states, 252)* | 10.67% | 16.53% | 0.697 | -37.65% | -2.48% |
| HRP (756) | 10.79% | 16.78% | 0.695 | -38.13% | -2.51% |
| *Soft-regime conditional HRP (756)* | 10.57% | 16.68% | 0.686 | -37.71% | -2.50% |
| Equal weight | 11.11% | 17.92% | 0.678 | -39.22% | -2.66% |
| *Dynamic graph-temporal, 5-seed median* | — | — | 0.681 | — | — |

Italicised rows are the frozen candidates. The dynamic model's five prescribed
seeds give Sharpe 0.598 / 0.615 / 0.681 / 0.701 / 0.723 (median 0.681, spread
0.126).

Both regime candidates land between the weakest and strongest baselines. The
dynamic model's median is below every baseline except equal weight.

## Paired inference

21-session moving block, 5,000 replications, seed 7, on matched dates only.
Each candidate is compared against equal weight and against the strongest
baseline on this panel. The strongest baseline was identified from the
baselines alone, so no candidate could select its own easiest comparator.

| Candidate | vs equal weight | | vs partial-corr/CVaR | |
| --- | ---: | ---: | ---: | ---: |
| | Δ ann. return | p | Δ ann. return | p |
| Soft-regime conditional HRP | -0.71% | 0.889 | +0.02% | 0.485 |
| HMM conditional risk (2/252) | -0.64% | 0.851 | +0.10% | 0.451 |
| Dynamic, seed 7 | +0.20% | 0.428 | +1.14% | 0.236 |
| Dynamic, seed 17 | -0.37% | 0.645 | +0.58% | 0.339 |
| Dynamic, seed 29 | -1.35% | 0.889 | -0.40% | 0.600 |
| Dynamic, seed 43 | -1.23% | 0.887 | -0.29% | 0.570 |
| Dynamic, seed 59 | +0.76% | 0.226 | +1.70% | 0.131 |

No comparison reaches the pre-registered 5% threshold. The smallest p-value in
the table is 0.131.

**Every candidate has a negative Sharpe differential against the strongest
baseline**, including the two whose mean return is higher. Seed 59 is the
clearest case: +1.70% annualised mean return against partial-correlation/CVaR
but a Sharpe differential of -0.025, because it carries the extra return on
materially more volatility. Reporting either statistic alone would misdescribe
it, which is precisely the substitution the protocol was written to prevent.

## Multiplicity-aware inference

White Reality Check over the complete seven-candidate confirmation family,
21-session blocks, 5,000 replications, on the 3,120 common dates:

| Benchmark | Best candidate | Best ann. excess | Family-wise p |
| --- | --- | ---: | ---: |
| Equal weight | dynamic seed 59 | +0.76% | **0.718** |
| Partial-correlation hub-avoid + mean-CVaR | dynamic seed 59 | +1.70% | **0.344** |

The family-wise result is unambiguous. Even the mean-return angle, which is
the candidates' most favourable framing, fails once the seven searched
candidates are priced in.

## The finding worth carrying into the paper

**The baseline ranking inverts between panels.** On the Track-A development
panel, equal weight was the strongest simple comparator (0.718) and the
graph/CVaR controls trailed it. On this panel the order reverses: the
graph/CVaR controls lead (0.748, 0.723) and equal weight is last (0.678).

Neither panel is unrepresentative, and nothing about the two universes was
chosen to produce this. It means a study that fixes one favourite baseline can
manufacture either a win or a loss through that choice alone, before any model
is trained. This argues for reporting a baseline *ladder* rather than a
baseline, and it is a stronger version of the point the protocol-inflation
table makes about reporting conventions.

Seed dispersion is also panel-dependent here: 0.126 Sharpe spread on this panel
against 1.079 on the 50-ETF development panel, from the identical model and
allocator. See [`PROTOCOL_INFLATION_RESULTS.md`](PROTOCOL_INFLATION_RESULTS.md).

## Secondary window (2020-01-31 boundary)

The amendment declared this window for comparability with the Track-A
chronological results. It is a reported robustness view, not a second chance to
select a candidate; its ledger is `selection_permitted: false`.

| Strategy | Ann. return | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| **Dynamic graph-temporal, seed 7** | 15.69% | 21.03% | **0.799** | -36.82% |
| Dynamic graph-temporal, seed 59 | 14.42% | 21.17% | 0.743 | -38.91% |
| Partial-correlation hub-avoid + mean-CVaR | 12.24% | 17.85% | 0.737 | -34.29% |
| Inverse volatility | 13.38% | 20.27% | 0.722 | -38.33% |
| Dynamic graph-temporal, seed 17 | 13.61% | 20.98% | 0.714 | -36.53% |
| Equal weight | 13.63% | 21.15% | 0.711 | -39.22% |
| HMM conditional risk (2/252) | 12.64% | 19.41% | 0.711 | -37.65% |
| HRP (252) | 12.63% | 19.38% | 0.711 | -37.69% |
| Soft-regime conditional HRP | 12.28% | 19.63% | 0.689 | -37.71% |
| HRP (756) | 12.19% | 19.81% | 0.680 | -38.13% |
| PCC hub-avoid + mean-CVaR | 11.12% | 18.01% | 0.676 | -35.98% |
| Dynamic graph-temporal, seed 43 | 11.82% | 20.91% | 0.640 | -37.05% |
| Dynamic graph-temporal, seed 29 | 11.17% | 21.70% | 0.597 | -39.85% |

### This window is the paper's sharpest case study

On this window, and only on this window, a candidate leads the table. Seed 7
reaches 0.799 against the strongest baseline's 0.737, with +3.78% annualised
mean return and a paired one-sided **p = 0.065** — precisely the kind of number
a paper describes as "marginally significant" and builds a claim around.

Every honest reading of the *same runs* removes it:

| Reading | Result |
| --- | --- |
| Seed 7 alone | Sharpe 0.799, beats every baseline, p=0.065 |
| Five-seed median | Sharpe 0.714, **below** the best baseline's 0.737 |
| Worst prescribed seed | Sharpe 0.597, below all six baselines |
| White Reality Check over the seven-candidate family | **p = 0.185** |
| Deflated Sharpe over all thirteen trials | **0.513** |

The seed spread on this window is 0.202. The deflated Sharpe is the most
direct statement: observed best daily Sharpe 0.0503 against an expected
maximum of 0.0495 under the null across thirteen trials, giving 0.513. The
leading result is what searching thirteen strategies produces with no skill.

No selection is made from this window, and seed 7 is not promoted. The
pre-registered conclusion stands on the primary window. This section is
reported because the near-miss is the most instructive artefact the study
produced: it is a complete, pre-registered demonstration that the difference
between a publishable claim and no effect can be entirely a matter of which
reading of the same runs is reported.

## Scope and limitations

This panel is a fixed US-listed ETF allocation universe, not a point-in-time
index-constituent study. Both panels use the same provider (`yfinance`) and
overlap in calendar time, so this is an independent *instrument* set rather
than an independent data source or an independent market history. Confirmation
here does not discharge the membership, delisting, identity-continuity, or
total-return requirements in charter v1 §4.

`paper_eligible` remains false for every run in this cycle.

## Schema change and full re-execution — resolved

The candidate-ledger validator originally required a concrete
`last_signal_date` and rejected the open-ended window this amendment declared.
It was extended to accept an explicit null end date, with the audit requiring
the ledger and the runs to agree in **both** directions. That change was made
after the first pass of runs completed.

Rather than rely on a gate touched post hoc, the entire confirmation cycle was
re-executed with **concrete window bounds**: both ledgers now declare
`last_signal_date: 2026-07-24` and every run is launched with an explicit
`--last-signal-date`. The validator's open-ended path is therefore never
exercised by any reported result, and no number in this document depends on the
schema change. The extension remains in the codebase as a tested improvement
that nothing here relies on.

### The re-execution reproduced every run exactly

All 26 runs across both windows are bit-identical to the originals — maximum
absolute daily-return difference **0.0**, including all ten stochastic dynamic
model fits. The complete downstream chain was regenerated against the new
manifests and every derived statistic is unchanged:

| Quantity | Value | Reproduced |
| --- | ---: | :---: |
| Primary Reality Check vs equal weight | 0.7181 | yes |
| Primary Reality Check vs strongest baseline | 0.3437 | yes |
| Primary deflated Sharpe (13 trials) | 0.4844 | yes |
| Secondary Reality Check vs equal weight | 0.6081 | yes |
| Secondary Reality Check vs strongest baseline | 0.1852 | yes |
| Secondary deflated Sharpe (13 trials) | 0.5126 | yes |
| Five-factor alphas | 0/13 positive, none significant | yes |
| Protocol-inflation table | all twelve rows | yes |

The dynamic graph-temporal model is fully deterministic given its seed, at a
later code revision and under a changed window specification. That is a
stronger reproducibility statement than the protocol required, and it is
reported here because a frozen protocol that did *not* reproduce would have
been a finding in its own right.

## Provenance

- Ledger: `configs/ledgers/public_us_etf_alternative_confirmation_v1.json` (7/7 complete)
- Paired inference: `results/confirmation/confirmation_inference_summary.json`
- Reality Checks: `results/confirmation/alt_etf_confirmation_reality_check_*.json`
- Run directories: `results/canonical/alt-etf-confirm-*` (13 runs)
