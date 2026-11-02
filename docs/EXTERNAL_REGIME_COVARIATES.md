# Public external regime-covariate pipeline

This pipeline adds observed market-state covariates without accidentally
turning them into investable portfolio assets. It is intended for transparent
regime inference, not for a claim that Yahoo Finance provides an institutional
macro data feed.

## Contract

`scripts/download_public_ohlcv.py --allow-non-tradable-price-series` permits
an index-style series with constant or zero volume only when it is explicitly
kept outside the membership file. `scripts/build_price_regime_covariates.py`
then produces the close level, one-session return, 21-session trailing return,
and realised volatility. No value is filled forward. The dynamic runner accepts
the resulting `date + numeric columns` CSV via `--regime-covariates`, joins it
on exact date only, verifies finite values, and hashes both the data and its
source sidecar in the canonical run manifest.

Available local, provenance-recorded packages are:

| Market | Non-tradeable source | Raw dates | Causal covariate dates |
| --- | --- | ---: | ---: |
| US | `^VIX` | 4,165 | 4,144 |
| India | `^INDIAVIX` | 4,065 | 4,044 |

Both remain outside their respective ETF universes. The public data package is
usable for diagnostics but does not solve the absence of point-in-time stock
membership or macro-release timestamp data required for a final equity paper.

## US VIX development diagnostic

The VIX candidate adds `vix_level`, `vix_one_session_return`, and
`vix_realized_volatility` to the expanding three-state soft K-Means regime
features. The direct-probability dynamic model was run at five epochs and all
five prescribed seeds. Because the causal regime/sequence warm-up delays the
first target, every comparator is restricted to 2016-05-02 through 2019-12-31
(924 realised daily observations and 10 bps one-way costs).

| Variant | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Regime-free dynamic model, five-seed mean | 9.04% ± 1.53% | 9.12% ± 0.97% | 0.989 ± 0.076 | -15.26% ± 1.63% |
| VIX-conditioned regime model, five-seed mean | 8.45% ± 1.07% | 8.75% ± 0.92% | 0.975 ± 0.110 | -12.03% ± 1.55% |

The seed-average paired annualised mean-return difference is -0.57% against
the regime-free model (21-session moving block, 5,000 replications,
one-sided `p=0.717`). VIX reduces average maximum drawdown but does not improve
the primary return or Sharpe outcome, so it is not advanced to outer evaluation.

The result is informative rather than wasted work: the pipeline can now test
open market-state data reproducibly in both markets, and it demonstrates that
adding one volatility index is not itself a sufficient contribution.
