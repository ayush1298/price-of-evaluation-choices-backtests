# Rolling baseline matrix results

This study replaces a single favourable split with a fixed, repeated
chronological stability check for the 30-ETF US public panel. It evaluates
only conventional baselines; it does not tune a method per period and it is
not an additional source of neural-model hyperparameter selection.

## Frozen protocol

`configs/public_us_etf_2010_rolling_baseline_v1.json` fixes six non-overlapping
two-calendar-year windows from 2014 through 2025 and four strategies:

1. PCC + all assets + equal weight;
2. PCC + all assets + inverse volatility;
3. PCC + all assets + Hierarchical Risk Parity; and
4. PCC + hub avoidance + mean-CVaR.

`scripts/run_rolling_baseline_matrix.py` makes each monthly signal with all
prior data available, but starts the reported backtest at the window's first
signal and truncates it at the final signal. It uses the fixed public US
configuration: 252-session lookback, 10% cap, one-session execution lag, and
10 bps one-way costs. Therefore no future date in a fold can affect a signal,
target, cost, or performance observation in that fold. The canonical output
contains every fold's net daily returns, graph diagnostics, metrics, and a
manifest of both configuration files and input hashes.

## Results

Each fold has 481--484 realised sessions. The table reports net annualised
Sharpe; bold values identify the best conventional strategy *within* that
fold, not a model-selection target.

| Fold | Equal weight | Inverse volatility | HRP | PCC hub-avoid + mean-CVaR |
| --- | ---: | ---: | ---: | ---: |
| 2014--2015 | 0.318 | **0.362** | 0.230 | 0.351 |
| 2016--2017 | 1.908 | 2.075 | 2.122 | **2.166** |
| 2018--2019 | 0.490 | 0.662 | 0.722 | **0.900** |
| 2020--2021 | 0.772 | 0.748 | **0.803** | 0.729 |
| 2022--2023 | **0.227** | 0.164 | 0.116 | -0.234 |
| 2024--2025 | 1.264 | 1.298 | **1.320** | 1.207 |
| Unweighted fold mean | 0.830 | 0.885 | **0.886** | 0.853 |
| Fold Sharpe standard deviation | 0.647 | 0.700 | 0.745 | 0.812 |

The average is descriptive, not a pooled performance statistic. It does not
license selecting HRP: its advantage over inverse-volatility is only 0.001
Sharpe on an unweighted fold average, and its fold volatility is higher. Two
strategies win two folds each (HRP and PCC/CVaR); equal weight and
inverse-volatility win one each. The evidence therefore supports reporting a
strong baseline set and regime dependence, not presenting any conventional
allocation rule as universally dominant.

The implication for the proposed model is concrete: it must be frozen before
evaluation and beat this matrix on a predeclared aggregate criterion, with
paired inference over its aligned daily returns. A single better 2020+ Sharpe
or one attractive regime is insufficient for an ICAIF claim.
