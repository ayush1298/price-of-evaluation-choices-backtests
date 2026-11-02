# Cross-sectional information-coefficient objective diagnostic

This document records the optional training objective that combines the
existing masked Gaussian negative log likelihood with `0.1 × (1 - IC)`. IC is
the differentiable Pearson correlation between the asset forecast scores and
their completed next-rebalance returns within each eligible cross-section.
It uses only the same forward labels already permitted in expanding model
training; it neither changes signal-date inputs nor the allocator.

The frozen implementation is in
`DynamicGraphTransformer.cross_sectional_information_coefficient_loss`, with
market configurations in `configs/public_*_information_coefficient_dynamic_graph.json`.

## US development evidence

All results below use the pre-2020 development period 2014-02-03 to 2019-12-31,
10 bps one-way costs, five epochs, and seeds `[7, 17, 29, 43, 59]`.

| Variant | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Gaussian likelihood only | 7.35% ± 1.11% | 9.78% ± 1.29% | 0.775 ± 0.071 | -15.42% ± 1.79% |
| Gaussian likelihood + IC objective | 7.59% ± 1.54% | 9.84% ± 0.97% | 0.788 ± 0.104 | -15.94% ± 1.85% |

The equally seed-averaged daily-return diagnostic has an annualised excess
return of +0.23% and a 21-session, 5,000-replication paired moving-block
one-sided p-value of 0.298. This is a small, statistically inconclusive
development change, not a basis for a US outperformance claim.

## India cross-market diagnostic

The IC weight was fixed from the US development candidate and evaluated on the
same India outer period as the existing regime-free dynamic model
(2020-02-03 to 2026-07-24; 1,577 realised INR daily observations). It fails
to transfer:

| Variant | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Gaussian likelihood only, five-seed mean | 19.20% ± 0.72% | 14.36% ± 0.55% | 1.297 ± 0.063 | -20.91% ± 1.03% |
| Gaussian likelihood + IC objective, five-seed mean | 18.09% ± 1.05% | 14.39% ± 0.53% | 1.230 ± 0.090 | -21.42% ± 1.56% |

The seed-averaged paired annualised mean-return difference is -0.94% against
the regime-free dynamic model. Its 95% moving-block interval is entirely
negative and the one-sided probability of a positive advantage is 0.998.
The IC objective is therefore retained as a reported failed candidate and is
not advanced to an outer US run or proposed-method status.

The wider inference is useful: a ranking-aligned loss alone does not overcome
the limited cross-sectional signal in these public fixed-ETF panels. The next
paper-quality model selection should be done on a broader independent universe
or with more informative non-price features, not by repeatedly searching this
same diagnostic test set.
