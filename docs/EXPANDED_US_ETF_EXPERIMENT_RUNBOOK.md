# Expanded fixed-US-ETF public experiment

This is a separate, no-licence robustness panel for the regime-aware portfolio
research. It has a wider fixed universe, not a historical S&P 500 constituent
history. Every one of the 50 US-listed ETFs in
`configs/universes/us_etf_2010_expanded_v1.csv` traded before 2010 and was
declared before extraction; it spans broad, country, sector, credit, duration,
real-estate, commodity, precious-metal, and thematic exposures.

## Data gate

Yahoo Finance via `yfinance` supplied adjusted OHLCV through the existing
provenance-recorded downloader. The local package passes the strict market-data
validator with these properties:

| Property | Value |
| --- | ---: |
| Date range | 2010-01-04 to 2026-07-24 |
| Assets | 50 |
| Genuine OHLCV rows | 208,200 |
| Monthly membership snapshots | 199 |
| Minimum assets per snapshot | 50 |
| Membership staleness | 0 days |

It is still a fixed-ETF allocation study and must not be described as a
point-in-time equity-index study or a survivorship-free stock-selection result.

## Outer baseline diagnostic

The first 2020-01-31-boundary runs have 1,627 realised daily observations and
charge 10 bps one-way costs.

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 11.28% | 15.70% | 0.760 | -30.05% |
| Inverse volatility | 8.51% | 11.36% | 0.776 | -21.74% |
| PCC hub-avoid + mean-CVaR | 2.18% | 6.81% | 0.351 | -16.13% |
| Partial-correlation hub-avoid + mean-CVaR | 2.69% | 6.65% | 0.432 | -16.23% |

The broader panel changes the allocation trade-off: the graph/CVaR controls
greatly reduce downside exposure but fail to retain enough return. Equal and
inverse-volatility are therefore mandatory primary comparators for this panel.

## Dynamic-model development gate

The unchanged causal graph-temporal model was evaluated only through
2019-12-31 at five epochs and five prescribed seeds. Its seed dispersion is
large, so a best-seed result is explicitly excluded from selection.

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 6.32% | 10.00% | 0.663 | -19.23% |
| Inverse volatility | 5.80% | 6.57% | 0.891 | -12.35% |
| PCC hub-avoid + mean-CVaR | 4.31% | 3.17% | 1.347 | -5.36% |
| Partial-correlation hub-avoid + mean-CVaR | 4.00% | 3.08% | 1.289 | -5.12% |
| Dynamic model, five-seed mean | 6.95% ± 4.28% | 9.75% ± 1.43% | 0.771 ± 0.464 | -17.94% ± 8.26% |
| Equal target-weight dynamic seed ensemble | 7.24% | 8.07% | 0.907 | -13.65% |

The best seed has Sharpe 1.261 (seed 59) with seed 7 close behind at 1.192, and
positive paired mean-return evidence against the simple controls, but another
prescribed seed has Sharpe 0.182 (seed 29) and a -29.36% drawdown. The five
prescribed seeds span 1.079 Sharpe with a median of 0.759; the identical
configuration, data, and allocator differ only in initialisation. That
instability is disqualifying, and it is quantified as a reporting shortcut in
[`PROTOCOL_INFLATION_RESULTS.md`](PROTOCOL_INFLATION_RESULTS.md). A strict five-member
target ensemble was generated with `scripts/ensemble_canonical_targets.py`;
all members have identical signals/assets, and it is priced only through the
development endpoint. It improves stability but not enough to beat the
low-volatility graph/CVaR controls on the primary Sharpe metric. It is not
advanced to the 2020+ dynamic outer run.

This panel is retained for future validation once a seed-stable model and an
ex-ante ensemble rule have been selected on an independent development period.
