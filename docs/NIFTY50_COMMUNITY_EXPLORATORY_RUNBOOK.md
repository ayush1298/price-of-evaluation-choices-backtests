# Community Nifty 50 Membership + Yahoo Prices: Exploratory Runbook

## Scope

This package is a stock-level **exploratory development** dataset, not a
paper-ready Nifty 50 backtest.  It combines a community-published historical
weight reconstruction with public Yahoo Finance prices.  The two components
are useful precisely because they expose historical membership changes, but
they do not yet meet the project’s required standards for independently
audited membership, ticker identity, delisting returns, or corporate-action
reconciliation.

All runs using `configs/exploratory_nifty50_community_yahoo_2008_monthly.json`
are marked `paper_eligible=false` and cannot enter a standard paper table.

## Source audit

Kaggle source: `megachonk/historical-nifty-50-constituent-weights-20y`, version
4, retrieved 2026-07-26.  Its `weights.csv` SHA-256 is
`3a3441a997b64a54363b38ebc02807e82cc0e0affd2efda147d1b7758a0b10cb`.
The source states that it reconstructs monthly and event-driven history from
Nifty Indices reports, but this assertion has not been independently reconciled
against official notices.  It is CC BY-NC-SA 4.0, so any academic use must
follow its attribution/non-commercial/share-alike terms.

The strict importer verifies 216 dated snapshots from 2008-01-31 through
2025-08-31, exactly 50 positive-weight constituents per snapshot, 100 unique
tickers across the source, and a 100.000000% sum on every snapshot.  Positive
weight is converted to `eligible=true`; the last available snapshot is carried
forward to the next snapshot by the canonical point-in-time lookup.  Source
weights are retained as `index_weight_pct` only for audit and are not used by
the present equal-weight/graph/CVaR strategies.

## Public-price coverage audit

Yahoo Finance was queried for all 100 source tickers with `auto_adjust=true`
from 2008-01-01 through 2025-09-01.  The saved long OHLCV panel has 348,585
rows for 83 tickers through 2025-08-29.  Sixteen requested symbols produced no
usable Yahoo history:

`BAJAJAUTO`, `CAIRN`, `HDFC`, `IBULHSGFIN`, `IDFC`, `INFRATEL`, `JPASSOCIAT`,
`LTIM`, `MCDOWELL-N`, `RANBAXY`, `RPL`, `SATYAMCOMP`, `SESAGOA`, `STER`,
`TATAMOTORS`, and `VSNL`.

`RELCAPITAL` was explicitly excluded because Yahoo returned a materially
inconsistent OHLC bar on 2009-06-29 (the stated high was below the close and
low).  No missing price or bar has been filled, rewritten, or back-adjusted by
this project.  The Yahoo source sidecar contains the full requested, downloaded,
missing, and invalid ticker lists plus the provider-to-canonical ticker mapping.

Against the 216 membership snapshots, 212 monthly decision dates have between
40 and 48 priced eligible constituents (median 45); no date has all 50.  The
source snapshots are at most 35 calendar days old at those decision dates.
Thus `minimum_eligible_assets=40` is an exploratory coverage threshold, not a
claim of complete Nifty 50 replication.

## Reproducible build

```bash
python -B scripts/import_kaggle_nifty50_membership.py \
  --weights data/exploratory/kaggle_nifty50_weights_v4/raw/weights.csv \
  --dataset-version 4 \
  --output data/exploratory/kaggle_nifty50_weights_v4/membership_snapshots.csv

python -B scripts/download_public_ohlcv.py \
  --symbols data/exploratory/kaggle_nifty50_weights_v4/yahoo_symbols_ns.csv \
  --symbol-column symbol --output-symbol-column nifty_ticker \
  --start 2008-01-01 --end 2025-09-01 \
  --allow-missing-symbols --drop-symbols-with-invalid-ohlcv \
  --output data/exploratory/kaggle_nifty50_weights_v4/ohlcv_yfinance_ns.csv

python -B scripts/validate_market_dataset.py \
  --prices data/exploratory/kaggle_nifty50_weights_v4/ohlcv_yfinance_ns.csv \
  --universe data/exploratory/kaggle_nifty50_weights_v4/membership_snapshots.csv \
  --market INDIA_NIFTY50_COMMUNITY_EXPLORATORY --currency INR \
  --rebalance-frequency ME --minimum-eligible-assets 40 \
  --maximum-membership-staleness-days 35 --require-ohlcv \
  --output data/exploratory/kaggle_nifty50_weights_v4/validation_report_min40.json
```

## Experiment protocol

Use 2014-01-01 through 2019-12-31 only for exploratory baseline/model
development.  Freeze any candidate before the 2020-01-01 through 2025-08-29
outer slice.  Every comparison uses the same dynamic membership history,
minimum 40 continuously priced eligible assets, monthly execution, one-session
lag, 10 bps one-way costs, and a 5% cap.

Any promising method must be rerun unchanged on an independently reconciled
Nifty membership series with delisting-aware prices before it is described as
an India result in the paper.

The matched graph-temporal Transformer pilot uses
`configs/exploratory_nifty50_community_yahoo_2008_dynamic_graph.json` and the
same five prescribed seeds.  Its master tensor universe is frozen from the
first declared model signal, so it is an ablation of the model architecture—not
a remedy for the source’s missing historical price identities.

The completed conventional and Transformer-feasibility evidence is recorded in
[the Nifty community exploratory results note](NIFTY50_COMMUNITY_EXPLORATORY_RESULTS.md).
The follow-up official exchange coverage test is recorded separately in [the
NSE bhavcopy coverage audit](NSE_BHAVCOPY_COVERAGE_AUDIT.md); it improves the
data-acquisition path but is deliberately not merged into this adjusted-Yahoo
experiment.
