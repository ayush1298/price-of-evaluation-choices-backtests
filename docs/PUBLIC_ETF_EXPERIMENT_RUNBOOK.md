# Public ETF experiment runbook

This is the no-licence empirical track. It is intentionally distinct from the
future point-in-time stock-universe track. Public raw downloads and canonical
run output remain local and are ignored by Git; manifests record their hashes.

## Scope

`configs/universes/us_etf_2010_v1.csv` freezes 30 liquid US-listed ETFs before
extraction. They span US equity, long-standing sector ETFs, international
equity, bonds, REITs, gold, and broad commodities. The research object is
therefore a **fixed-ETF allocation universe**, not an S&P 500 constituent
strategy.

This design permits an honest membership file: all assets are declared before
the sample starts and must have an observed close at each monthly decision
date. It must never be reused for an equity-index constituent claim.

## Reproducible acquisition

```bash
python -B scripts/download_public_ohlcv.py \
  --symbols configs/universes/us_etf_2010_v1.csv \
  --start 2010-01-01 --end 2026-07-25 \
  --output data/public/us_etf_2010_v1/ohlcv_yfinance.csv

python -B scripts/build_fixed_universe_membership.py \
  --prices data/public/us_etf_2010_v1/ohlcv_yfinance.csv \
  --symbols configs/universes/us_etf_2010_v1.csv \
  --output data/public/us_etf_2010_v1/membership_monthly.csv

python -B scripts/validate_market_dataset.py \
  --prices data/public/us_etf_2010_v1/ohlcv_yfinance.csv \
  --universe data/public/us_etf_2010_v1/membership_monthly.csv \
  --market US_ETF --currency USD --rebalance-frequency ME \
  --minimum-eligible-assets 25 --maximum-membership-staleness-days 0 \
  --require-ohlcv
```

The downloader uses `auto_adjust=True` and records the retrieval timestamp,
the `yfinance` version, source manifest hash, and output shape in an adjacent
`.source.json` file. The membership builder records its static-universe rule
and all source hashes in its own sidecar. Canonical experiment manifests
automatically include these sidecars when present.

## Current validated dataset

The first extraction, completed 2026-07-26, contains:

| Property | Value |
| --- | ---: |
| Date range | 2010-01-04 to 2026-07-24 |
| Assets | 30 |
| Genuine OHLCV rows | 124,920 |
| Monthly membership snapshots | 199 |
| Minimum assets observed at a snapshot | 30 |
| Membership staleness | 0 days |

## Experiment matrix

Run the graph/allocation baselines with
`configs/public_us_etf_2010_monthly.json`:

1. equal weight;
2. inverse volatility;
3. PCC + hub avoidance + mean-CVaR; and
4. shrinkage partial correlation + hub avoidance + mean-CVaR.

Run the causal probabilistic HMM mixture with
`configs/public_us_etf_2010_regime_hmm.json` over all seeds
`[7, 17, 29, 43, 59]`. Run graph-temporal, graph-only, and temporal-only
ablation configurations with their corresponding `public_us_etf_2010_*.json`
files. Each model decision must train only on forward labels completed before
that decision date.

## Initial empirical evidence (not final paper claims)

These results use 10 bps one-way transaction costs and are stored only in
local manifest-backed run directories:

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 8.88% | 13.71% | 0.689 | -30.87% |
| Inverse volatility | 8.07% | 11.55% | 0.730 | -27.30% |
| PCC hub-avoid + mean-CVaR | 6.15% | 7.72% | 0.812 | -18.16% |
| Partial-correlation hub-avoid + mean-CVaR | 6.68% | 7.87% | 0.861 | -18.40% |
| HMM regime mixture, five-seed mean | 5.35% | 8.80% | 0.636 | -18.50% |

The partial-correlation strategy has the best risk-adjusted return among the
completed baselines, but its matched mean daily return is lower than equal
weight. A 5,000-replication 21-session paired moving-block bootstrap does not
support a positive mean-return advantage over equal weight. This is evidence
for downside-risk control, not an alpha claim.

The 20-epoch DeepCNL graph baseline was also trained from 2010-01-04 through
2013-12-31 using all 30 ETFs and genuine OHLCV. With its pre-declared 5% rare
edge rule, it has 30 nodes and 21 learned edges. This validates the preserved
model's input path; it is not yet a rolling portfolio comparison.

### Dynamic-model development evidence

All dynamic results begin in 2014 because the causal model first requires 24
completed forward-return windows. Therefore compare them only on matched dates,
not against full-period baseline headline metrics.

| Variant | Training budget | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Full graph-temporal, seed 7 | 5 epochs | 0.748 | -23.43% |
| Graph-only, seed 7 | 1 epoch | 0.660 | -20.40% |
| Temporal-only, seed 7 | 1 epoch | 0.577 | -26.05% |
| Full graph-temporal, five-seed mean | 1 epoch | 0.703 ± 0.055 | -29.38% ± 4.95% |

At the development budget, full graph-temporal training has a positive matched
mean-return differential against graph-only (block-bootstrap one-sided
`p=0.030`), while the temporal-only comparison is not conventionally
significant (`p=0.098`). Against the simple partial-correlation/CVaR baseline,
the five-epoch full model has lower matched Sharpe and only borderline positive
mean-return evidence (`p=0.052`). These are diagnostic findings, not an
outperformance claim.

Any paid-data extension must rerun this entire registered matrix rather than
transfer these headline metrics.

## Final outer evaluation protocol

The results above are development evidence: they informed the choice of the
partial-correlation baseline and exposed that the HMM mixture is not yet a
winner. They must not be presented as a final held-out test. Before the paper
table is produced, freeze the configuration and code revision, then evaluate
with the inclusive signal boundary `2020-01-31`. Observations before that date
remain available as history (and, for the dynamic model, as completed training
labels); no allocation is backtested before the boundary. The date, horizon,
cost model, seeds, and primary metric must not change after seeing those
results.

Every canonical runner now accepts `--first-signal-date`, for example:

```bash
python -B scripts/run_canonical_baseline.py \
  --run-id public-etf-outer-partial-cvar \
  --prices data/public/us_etf_2010_v1/ohlcv_yfinance.csv \
  --universe data/public/us_etf_2010_v1/membership_monthly.csv \
  --config configs/public_us_etf_2010_monthly.json \
  --graph partial_correlation --selector hub_avoid --allocator mean_cvar \
  --first-signal-date 2020-01-31

python -B scripts/run_canonical_dynamic_graph_model.py \
  --run-id public-etf-outer-dynamic-seed7 \
  --prices data/public/us_etf_2010_v1/ohlcv_yfinance.csv \
  --universe data/public/us_etf_2010_v1/membership_monthly.csv \
  --config configs/public_us_etf_2010_dynamic_graph.json --seed 7 --epochs 5 \
  --first-signal-date 2020-01-31
```

Use the identical boundary for all comparators and report their matched daily
period. A later paid-data study should preserve this protocol, while replacing
only the validated price and historical-membership inputs.

### First chronological results (diagnostic, not a paper headline)

The first runs under this boundary cover 2020-02-03 through 2026-07-24 (1,627
matched daily observations), including 10 bps one-way transaction costs:

| Strategy | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | 10.70% | 15.92% | 0.718 | -30.87% |
| Partial-correlation hub-avoid + mean-CVaR | 5.69% | 9.72% | 0.619 | -18.40% |
| HMM regime mixture, five-seed mean | 5.70% | 10.15% | 0.597 ± 0.003 | -18.20% |
| Full graph-temporal model, five-seed mean | 9.27% | 14.45% | 0.685 ± 0.058 | -26.72% |

For seed 7, the full model (Sharpe 0.730) exceeds its matched graph-only
(0.431) and temporal-only (0.551) ablations. Its paired block-bootstrap
mean-return differential is positive against both ablations (one-sided
`p=0.007` and `p=0.023`) and against the partial-correlation baseline
(`p=0.038`). Its mean return does **not** exceed equal weight (`p=0.705`).
The five full-model seeds range from 0.590 to 0.734 Sharpe. Their average
Sharpe remains below equal weight (0.718), and their drawdowns are variable;
there is no current outperformance claim.

### Factor-exposure robustness

The public Kenneth French daily five-factor archive is downloaded by
`scripts/download_fama_french_daily_factors.py` and stored with its archive
hash and retrieval metadata. `scripts/run_factor_evaluation.py` aligns a
canonical run's daily **net** returns with the factor dates, subtracts the
daily `RF`, and estimates OLS factor loadings with five-lag Newey-West alpha
standard errors. The alpha below is a linear daily-alpha × 252 convention,
not a compounded portfolio return. All models have 1,589 matched daily dates.

| Strategy | Annualised factor alpha | HAC two-sided p-value | Market beta | R² |
| --- | ---: | ---: | ---: | ---: |
| Equal weight | -1.73% | 0.382 | 0.722 | 0.915 |
| Partial-correlation hub-avoid + mean-CVaR | -2.43% | 0.311 | 0.380 | 0.649 |
| HMM mixture, five-seed mean | -2.57% ± 0.03% | 0.255–0.268 | 0.402 | 0.689 |
| Full graph-temporal, five-seed mean | -1.41% ± 0.88% | 0.374–0.893 | 0.612 | 0.787 |

There is no evidence of positive five-factor alpha for the present public-ETF
study. This is a valuable negative result: future model work must improve
strictly held-out, factor-adjusted performance rather than optimise a
full-sample Sharpe ratio.

### Transaction-cost robustness

`scripts/reprice_canonical_targets.py` reprices an immutable source run's
stored targets under a new one-way cost assumption, preserving its dates,
weights, execution lag, and cap. It does not refit the graph, regime model, or
allocator. The table includes 78 monthly execution decisions and the
five-seed mean for the dynamic model.

| One-way cost | Equal-weight Sharpe | Partial-correlation/CVaR Sharpe | Dynamic five-seed Sharpe |
| ---: | ---: | ---: | ---: |
| 0 bps | 0.720 | 0.635 | 0.731 ± 0.058 |
| 5 bps | 0.719 | 0.627 | 0.708 ± 0.058 |
| 10 bps | 0.718 | 0.619 | 0.685 ± 0.058 |
| 25 bps | 0.715 | 0.594 | 0.616 ± 0.059 |
| 50 bps | 0.710 | 0.552 | 0.500 ± 0.061 |

Average one-way turnover per monthly execution is 2.6% for equal weight,
13.3% for partial-correlation/CVaR, and 56.0% for dynamic seed 7. The dynamic
model is therefore less cost-robust than equal weight and cannot be presented
as implementable until its allocation is stabilised (for example, through
validation-selected turnover penalties and a no-trade band).

### No-trade-band diagnostic

The backtest now supports an explicit realised-turnover no-trade band and
records both requested and executed turnover plus every skipped decision. On
the same five dynamic target streams at 10 bps, a 10% / 25% / 50% band yields
mean Sharpe 0.682 / 0.696 / 0.689, compared with 0.685 without a band. The
25% band skips 77 of 394 seed-decision pairs and reduces mean executed turnover
from 54.3% to 51.0%; at 50 bps it raises dynamic mean Sharpe from 0.500 to
0.522.

To separate the band decision from the 2020+ returns, the all-history causal
seed-7 stream was restricted to 2014--2019 for development. The bounded
evaluator truncates the price panel at the final development signal, so the
development return stream ends on 2019-12-31 rather than carrying the final
holding into the future. This corrected run contains 1,489 daily observations.
At 10 bps, Sharpe is 0.821 / 0.825 / 0.920 / 0.808 for no band, 10%, 25%, and
50%, respectively. The 25% band therefore remains the development choice; it
skips 36 of 71 decisions and reduces mean executed one-way turnover from 39.8%
to 34.5%. Applied unchanged to all five 2020+ target streams, it yields 0.696
± 0.093 Sharpe, marginally above the unbanded 0.685 ± 0.058 but below equal
weight (0.718). Thus the band is a modest implementability control, not a
performance solution.

This boundary enforces chronology in the code, but it was introduced after
development exploration used this dataset. Therefore these are diagnostic
chronological results, not a fully untouched test. A paper-quality final test
must use an independently held-out market or data arriving after the design is
frozen; it must retain the exact public-data protocol, seeds, costs, and
comparison rules above.

The later soft-regime-conditioned dynamic-model diagnostic is recorded in
[`REGIME_CONDITIONED_DYNAMIC_RESULTS.md`](REGIME_CONDITIONED_DYNAMIC_RESULTS.md).
It does not improve this regime-free model and remains an ablation rather than
a proposed-method result.

The completed causal graph-attention comparator is recorded in
[`GRAPH_ATTENTION_BASELINE_RESULTS.md`](GRAPH_ATTENTION_BASELINE_RESULTS.md).
It also fails the pre-2020 development gate, so it is a reported control rather
than an outer-study contender.

The ranking-aligned likelihood/IC objective is recorded in
[`INFORMATION_COEFFICIENT_MODEL_RESULTS.md`](INFORMATION_COEFFICIENT_MODEL_RESULTS.md).
Its small US-development gain does not transfer to India, so it is not selected
for an outer US run.

The wider 50-ETF public robustness panel and its non-selected dynamic-model
development result are recorded in
[`EXPANDED_US_ETF_EXPERIMENT_RUNBOOK.md`](EXPANDED_US_ETF_EXPERIMENT_RUNBOOK.md).

The causal VIX covariate data path and its non-selected US regime result are
recorded in [`EXTERNAL_REGIME_COVARIATES.md`](EXTERNAL_REGIME_COVARIATES.md).

The Hierarchical Risk Parity allocation control, including its development-gate
and wider-universe outcomes, is recorded in
[`HIERARCHICAL_RISK_PARITY_RESULTS.md`](HIERARCHICAL_RISK_PARITY_RESULTS.md).

The fixed rolling two-year baseline matrix, which establishes the
cross-regime stability standard for later model selection, is recorded in
[`ROLLING_BASELINE_MATRIX_RESULTS.md`](ROLLING_BASELINE_MATRIX_RESULTS.md).

The soft-regime conditional covariance/HRP diagnostic and its matched
unconditional-risk control are recorded in
[`SOFT_REGIME_CONDITIONAL_RISK_RESULTS.md`](SOFT_REGIME_CONDITIONAL_RISK_RESULTS.md).
