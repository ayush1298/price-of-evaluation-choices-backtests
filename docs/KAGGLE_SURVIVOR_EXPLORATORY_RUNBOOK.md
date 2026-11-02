# Kaggle Current-S&P Survivor Panel: Exploratory Runbook

## Status and permitted use

This is a long-horizon **exploratory model-development panel**.  It is not a
historical S&P 500 constituent panel and is not eligible for a paper headline,
index-replication claim, or primary outperformance table.  Kaggle's source
contains the current company roster, so firms removed from the index are
absent.  Its current-company `date_added` field is only used as an additional
filter; it is not a permanent-identifier or delisting-history solution.

Use it to develop models, look for sensitivity to a wider stock cross section,
and decide whether a method merits a rerun on a point-in-time universe.  Keep
all such results visibly marked `exploratory_survivor_only_not_paper_eligible`.

## Acquired source and cleaning record

Source: Kaggle `darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily`,
version 118, retrieved on 2026-07-26.  The downloaded `sp500_stocks.csv` SHA-256
is `6e6ae2e144582e32526138472b0d1b2bbf662fe2449678d027ac9a4553120bb6`.

The raw file had 2,936,887 rows, 503 symbols, and dates from 2000-01-03 through
2026-07-24.  Its final date had missing OHLC values, so the importer drops that
entire date rather than creating an artificially sparse trading day.  One
vendor-inconsistent OHLC bar (HUBB, 2021-05-05, where the stated low exceeded
the open) led to exclusion of that entire symbol; no price values were repaired.
The resulting imported provider-close panel has 2,929,706 rows, 502 symbols,
and dates from 2000-01-03 through 2026-07-23.  The provider's close-adjustment
convention remains unverified, so these returns must not be described as total
returns or used to report alpha.

Raw, imported, and derived data live under `data/exploratory/` and are ignored
by git.  The importer source sidecar records hashes, scope, and every removal.

## Fixed 2010--2025 study cohort

The checked-in selection manifest is
`configs/universes/kaggle_current_sp500_survivor_entry_2010_v118.csv`.  It has
88 names: eight alphabetically selected, complete-history names from each of
the 11 current GICS sectors.  Each name has an observation on every available
market day from 2010-01-04 to 2025-12-31 and a current metadata `date_added`
not later than 2010-01-04.  The latter guard reduces obvious post-start ticker
reuse, but does not establish historical company identity.

Synchronizing that cohort without imputation yields 354,112 OHLCV rows across
4,024 fully observed daily dates, with 192 zero-staleness monthly membership
snapshots.  The exact build sequence is:

```bash
python -B scripts/import_kaggle_current_sp500.py \
  --stocks data/exploratory/kaggle_current_sp500_v118/raw/sp500_stocks.csv \
  --output data/exploratory/kaggle_current_sp500_v118/ohlcv.csv \
  --dataset-version 118 \
  --close-adjustment-convention 'provider_close_as_downloaded_adjustment_not_independently_verified' \
  --drop-incomplete-dates --drop-symbols-with-invalid-ohlcv

python -B scripts/build_kaggle_survivor_cohort.py \
  --prices data/exploratory/kaggle_current_sp500_v118/ohlcv.csv \
  --companies data/exploratory/kaggle_current_sp500_v118/raw/sp500_companies.csv \
  --start 2010-01-04 --end 2025-12-31 --per-sector 8 \
  --output data/exploratory/kaggle_current_sp500_v118/current_survivor_entry_2010_cohort_88.csv

python -B scripts/synchronise_fixed_universe_panel.py \
  --prices data/exploratory/kaggle_current_sp500_v118/ohlcv.csv \
  --symbols configs/universes/kaggle_current_sp500_survivor_entry_2010_v118.csv \
  --start 2010-01-04 --end 2025-12-31 \
  --output data/exploratory/kaggle_current_sp500_v118/ohlcv_cohort_88_2010_2025_synchronized.csv
```

## Experiment protocol

Use `configs/exploratory_kaggle_current_sp500_2010_monthly.json`.  Run a
baseline ladder during development (2015-01-01 through 2019-12-31): equal
weight, inverse volatility, HRP, then a constrained graph-selected CVaR
portfolio.  Freeze the chosen specification and run it once over the outer
period (2020-01-01 through 2025-12-31).  The protocol uses monthly rebalancing,
one-session execution lag, 10 bps one-way costs, an 88-name declared universe,
and a 5% single-name cap.

The config explicitly writes `paper_eligible=false` into baseline and dynamic
run manifests.  Standard canonical-table collection therefore excludes these
runs; internal inspection requires `include_exploratory=True`.

The completed conventional-baseline evidence is recorded in
[the exploratory results note](KAGGLE_SURVIVOR_EXPLORATORY_RESULTS.md).

## Required transition before paper use

Rerun any promising method against a source with point-in-time membership,
delisted-price treatment, stable identifiers, and a documented
split/dividend-adjustment convention.  The method, precommitted selection rule,
and evaluation windows should remain unchanged for that confirmation.
