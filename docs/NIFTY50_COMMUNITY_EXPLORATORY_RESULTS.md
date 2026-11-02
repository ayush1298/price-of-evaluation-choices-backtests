# Community Nifty 50 + Yahoo: Exploratory Results

## Non-paper status

These results use the partial community-membership/Yahoo panel documented in
[the Nifty runbook](NIFTY50_COMMUNITY_EXPLORATORY_RUNBOOK.md).  They are not a
Nifty 50 replication result, not eligible for a manuscript table, and must not
be combined with the fixed-ETF paper controls.  The result manifests carry
`paper_eligible=false`.

The source has valid dated constituent snapshots but only 40--48 observed
eligible Yahoo price series at each decision date; it has no full 50-stock
month.  All methods use the same monthly schedule, one-session lag, 10 bps
one-way costs, 5% cap, 252-session history, and at least 40 continuously
priced eligible assets.

## Conventional development ladder (2014--2019)

| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Daily CVaR 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 13.33% | 14.64% | 0.928 | -24.82% | -2.03% |
| Inverse volatility | 13.91% | 13.79% | 1.014 | -22.76% | -1.91% |
| HRP | 13.16% | 12.57% | 1.047 | -20.32% | -1.75% |
| PCC graph, hub avoidance, mean-CVaR (40 names) | 14.15% | 12.01% | 1.162 | -15.80% | -1.63% |
| Partial-correlation graph, hub avoidance, mean-CVaR (40 names) | 15.63% | 12.12% | 1.259 | -16.25% | -1.63% |

Partial-correlation/CVaR was the one frozen exploratory candidate because it
ranked first on development Sharpe.  Its paired 21-session moving-block test
against HRP was only borderline (annualized mean excess 2.11%, 95% interval
-0.46% to 4.66%, one-sided p=0.051).  Its five-trial deflated-Sharpe
diagnostic was 0.430.  This is insufficient to call the apparent development
gain an edge.

## Frozen outer diagnostic (2020--2025-08)

| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Daily CVaR 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal weight | 20.75% | 18.03% | 1.137 | -36.08% | -2.80% |
| HRP | 21.37% | 16.44% | 1.261 | -31.92% | -2.49% |
| Frozen partial-correlation/CVaR | 19.14% | 15.88% | 1.183 | -30.12% | -2.36% |

The frozen graph/CVaR candidate does not generalise.  Its annualized paired
mean return is -1.72% versus equal weight (one-sided p=0.729) and -1.95%
versus HRP (one-sided p=0.916).  It modestly reduces tail loss/drawdown, but
does not produce a return or risk-adjusted outperformance claim.

## Graph-temporal Transformer feasibility pilot

The dynamic Transformer initially exposed two corrected pipeline defects:

1. unbuildable graph warm-up dates were aborting a dynamic-universe run rather
   than being causally excluded and recorded; and
2. its tensor master universe was frozen at the first graph warm-up date,
   rather than the first declared model signal date.  That silently excluded
   legitimate first-signal constituents.

The fixed code records `graph_date_coverage.csv`, freezes the master universe
at the first model signal, and ends the daily backtest at the last actual model
target rather than carrying stale weights into a later period.  The corrected
one-epoch seed-7 smoke run freezes a 50-name tensor on 2014-01-31, has 120
buildable graph dates out of 132 candidates, and creates valid 40--43 asset
targets only from 2014-01-31 through 2016-03-31.  Its 528-day short-window
Sharpe is 1.154, but it is **not comparable** to the 2014--2019 ladder.

After March 2016, fewer than 40 names from the 2014-frozen causal tensor
remain simultaneously eligible and continuously priced.  Lowering the
minimum only for the neural model or admitting later assets would make the
comparison unfair or introduce future asset identities.  Consequently, no
multi-seed expansion, outer Transformer run, or model-selection claim is
permitted from this data source.

## Decision

The source provides a useful stress test for membership changes and has helped
correct dynamic-universe behavior.  It does not rescue the proposed method:
neither graph/CVaR nor the current graph-temporal model earns promotion.  The
next valid model experiment requires an audited point-in-time market source
with enough continuously priced historical identities to sustain the frozen
master universe throughout the development and outer windows.
