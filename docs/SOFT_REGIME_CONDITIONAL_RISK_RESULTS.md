# Soft-regime conditional covariance allocation diagnostic

This candidate moves regime information to the risk-estimation stage rather
than concatenating it to a neural return-forecast model. It was motivated by
the public-panel evidence that conventional risk controls are stronger and
more stable than the current return-forecast models.

## Candidate and causal contract

At each monthly signal close, an expanding three-state soft K-Means model is
fit only on earlier monthly market features and classifies the current close.
The four features are realised volatility, trailing market return, market
drawdown, and cross-sectional dispersion. Daily returns in the rolling history
are paired only with the latest state probability already known at that daily
close. The current probability vector then mixes state-weighted covariance
estimates. Each state covariance is shrunk 25% toward the unconditional
covariance and falls back to that unconditional estimate if it has fewer than
21 effective observations. The shared long-only, 10%-capped HRP allocator
then turns that covariance into target weights.

The implementation is in `src/research/experiments/regime_risk.py`; the runner
records every causal probability, target, execution, net daily return, and
conditional-covariance diagnostic in its canonical run directory. It has no
return forecast, future state label, forward-filled regime value, or changed
cost/backtest rule.

## US development gate

The first fixed configuration uses a 756-session covariance history and is
bounded at 2019-12-31 (1,489 realised daily observations, 10 bps one-way
costs). Its only purpose is to decide whether the regime mixture adds value
beyond an otherwise identical unconditional risk estimate.

| Method | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 7.84% | 10.19% | 0.792 | -16.08% |
| Inverse volatility, 252 sessions | 7.51% | 8.37% | 0.907 | -13.72% |
| HRP, 252 sessions | 6.59% | 7.31% | 0.910 | -13.05% |
| HRP, 756 sessions (matched control) | 6.94% | 7.30% | **0.956** | -11.90% |
| Soft-regime conditional HRP, 756 sessions | 6.56% | 7.48% | 0.886 | -12.93% |

Against the matched 756-session HRP control, conditional HRP has a -0.35%
annualised mean-return differential and a -0.070 Sharpe differential. A
5,000-replication, 21-session paired moving-block bootstrap gives a one-sided
positive-advantage p-value of 0.909. The regime covariance mixture therefore
does not pass the development gate and is not carried into 2020+ or India as a
proposed method.

## Stronger control, not a proposed result

The 756-session unconditional HRP control was selected by the development
comparison and run unchanged on the exploratory chronological outer windows.
It also fails to produce a universally dominant rule:

| Market/window | HRP 756 Sharpe | Relevant existing comparator | Comparator Sharpe |
| --- | ---: | --- | ---: |
| US 2020+ | 0.693 | Equal weight | 0.718 |
| India 2020+ | 1.300 | PCC hub-avoid + mean-CVaR | 1.313 |

It remains a stronger benchmark for later candidates, not the paper's
regime-aware contribution. The result reinforces the core conclusion: the
present open-data regime signal does not yet deliver a robust improvement once
compared with a well-specified conventional risk allocator.
