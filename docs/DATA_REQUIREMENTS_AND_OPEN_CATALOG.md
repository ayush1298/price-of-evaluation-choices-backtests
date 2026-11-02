# Data requirements and open-data catalogue

This document distinguishes three questions that are often conflated:

1. Can an input make the code run?
2. Can it support a valid backtest?
3. Can it support the claimed population of a paper?

An answer of yes to (1) is not automatically an answer of yes to (2) or (3).

## Required fields, purpose, and source options

| Input | Used by | Priority | Can public data supply it? | Paid/point-in-time requirement |
| --- | --- | --- | --- | --- |
| `date`, `symbol`, and daily adjusted close | returns, PCC/partial-correlation graphs, regimes, all allocators, daily NAV | **Essential** | Yes: `yfinance`, Yahoo download, Stooq, exchange files | No for a fixed ETF panel; yes when ticker history must be authoritative |
| Trading dates / exchange calendar | signal timing, next-session execution, annualisation | **Essential** | Yes, derive from the observed price panel; use official exchange calendars for an audit | No for the fixed ETF design |
| Open, high, low, close, volume (OHLCV) | legacy DeepCNL graph; optional liquidity/capacity checks | **Essential only for DeepCNL**; otherwise optional | Often yes from `yfinance` and exchange daily files | A licensed feed is preferable for stock-level production data; never fabricate these columns |
| Price-adjustment convention and corporate actions | correct returns across splits/dividends | **Essential** | Partly: `yfinance` can return an adjusted price series, but archive the download and convention | Strongly recommended for stock studies; robust event histories usually require a vendor |
| Point-in-time eligible membership (`date`, `symbol`, `eligible`) | prevents buying names before index entry or after removal; dynamic equity universe | **Essential for a changing company/index universe**; not needed for a fixed ETF list | Rarely reliable/free. Public community datasets can be audited candidates | Normally requires CRSP/Norgate/LSEG/FactSet, or an audited reconstruction |
| Delisting date, return, ticker/name changes, permanent ID | removes survivorship bias and ticker-reuse errors | **Essential for a changing company universe** | No reliable all-in-one free feed | Normally paid, or painstakingly reconstructed and documented |
| Rebalance-date index weights / market capitalisation | cap-weight benchmark, capacity rules, market-cap selection | Recommended, not needed for equal-weight/inverse-vol portfolios | Some community snapshots, but audit first | Paid or official index subscription for a defensible index-replication claim |
| Volume / dollar volume | liquidity screen, turnover/capacity robustness | Recommended | Yes for a fixed ETF panel and often for public equities | Point-in-time corporate-event consistency is better with a vendor |
| Benchmark/index or ETF return | market regime features and benchmark comparison | **Essential** for a credible evaluation | Yes: public index/ETF history | No for the ETF study |
| Risk-free rate | excess-return Sharpe/alpha | Recommended | Yes: FRED and the Kenneth French library | No |
| Factor and industry-portfolio returns | factor alpha, stress/regime validation, stronger baselines | Recommended | Yes: Kenneth French Data Library and other public academic libraries | No |
| Sector/industry labels | interpretation and sector-neutral tests | Optional | Current classifications are easy to obtain, but are not point-in-time | Paid/PIT source only if used as a historical model feature |
| Fundamentals, earnings, macro, news/sentiment | feature extensions | Optional | Macro is free; fundamentals/news coverage is partial | Point-in-time fundamentals/news usually paid |
| FX rates | a single base-currency global portfolio | Essential only for a cross-market combined portfolio | Yes for liquid FX series | No for per-market local-currency experiments |

## What `yfinance` can and cannot do

`yfinance` can download daily OHLCV for a declared ticker list, ETFs, index
proxies, FX proxies, and many currently tradable stocks. It is enough for:

- a fixed ETF universe;
- a fixed, pre-declared stock cohort whose composition is not claimed to track
  a changing index; and
- model development, ablations, and public-data replication experiments.

It cannot by itself provide the historical **membership**, **delisting return**,
or safe **ticker-identity** data required for an all-historical-members stock
portfolio. It is an unofficial Yahoo client and its own documentation says the
Yahoo API is intended for personal use; preserve the source terms, retrieval
date, raw download hash, and do not redistribute raw data.

The public adapter in `src/research/data/yfinance_source.py` uses genuine
downloaded OHLCV, omits missing rows rather than inventing values, and writes a
source sidecar through `scripts/download_public_ohlcv.py`.

## Audit of datasets already in this repository

| File(s) | Observed coverage | Verdict |
| --- | --- | --- |
| `data/prices-split-adjusted.csv` | 851,264 observations, 501 tickers, 2010-01-04 to 2016-12-30, O/H/L/C/volume | The original Kaggle NYSE panel. Suitable for a reproducible **legacy/prototype** graph or DeepCNL run after schema checks. It has no membership-history table, delisting treatment, permanent identifiers, or documented dividend-total-return convention, so it cannot alone support a changing-universe paper claim. |
| `data/prices_extended_historical.csv`, `data/prices_2017.csv`, `data/prices_extended_recent.csv` | close-only extensions covering 2006--2009, 2017, and 2018--2025 | Useful only for close-return prototyping. They must not be converted into synthetic OHLCV, and cannot retrain DeepCNL. They also inherit the fixed Kaggle ticker-list problem. |
| `data/SPY20000101_20171111.csv`, `data/SP500^GSPC20000101_20171111.csv`, `data/SPY_2006_2025.csv` | single benchmark series | Useful after parsing for market-regime and benchmark data; not an investable cross section. |

The Kaggle source is `dgawlik/nyse`. Its description identifies it as a
playground S&P 500 price/fundamental dataset spanning mostly 2010--2016; the
split-adjusted file corrects splits but does not supply historical constituent
snapshots. Treat it as a frozen benchmark, not a new evidence dataset.

## Open-data candidates worth considering

| Candidate | What it provides | Best use | Required audit before paper use |
| --- | --- | --- | --- |
| `yfinance` / Yahoo Finance | Daily OHLCV for declared instruments | **Primary no-cost fixed-ETF dataset** | Download date, missingness, adjustment convention, random cross-source checks, provider terms |
| NSE daily bhavcopy archives | Indian daily price/trade records, including many removed/delisted names | India price-identity and coverage reconstruction | Corporate-action adjustment, time-bounded ticker/ISIN crosswalk, delisting treatment, and a separate membership history |
| Kaggle `dgawlik/nyse` | US price panel + some historical fundamentals, 2010--2016 | Legacy reproduction only | Same as above; do not infer a point-in-time universe |
| Kaggle `darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily` | CC0 daily OHLCV for the **current** S&P 500 roster, from 2000 onward; current company/sector metadata and `date_added` | Long-horizon model development, stress testing, and explicitly survivor-only exploratory comparisons | Archive the exact dataset version/hash; establish adjustment convention; do **not** call it historical S&P 500 membership. Removed/delisted constituents are absent, and current `date_added` cannot repair that survivorship bias. |
| Kaggle `megachonk/historical-nifty-50-constituent-weights-20y` | User-published monthly Nifty 50 constituent/weight history from 2008, claimed survivorship-aware | Candidate **membership** feed for an India robustness study | Download the data card, inspect the construction source, reconcile random rebalance dates to official NSE records, map renamed/delisted tickers, and report gaps. It is not official data merely because it is on Kaggle. |
| Community S&P 500 historical-membership repositories | Daily or change-date membership histories | Prototype/audit input, not sole ground truth | Trace source lineage and reconcile a random sample to S&P announcements; handle ticker reuse and delisted prices |
| Kenneth French Data Library | Free daily/monthly factors and constructed research portfolios | Factor-alpha evaluation, regime validation, or a return-only robustness experiment | Use its documented return definitions; it has no individual-stock OHLCV, so DeepCNL is inapplicable |
| Public macro sources (FRED) | Rates and macro series | Optional exogenous regime features / risk-free rate | Publication timing and revisions if used as live features |

## Data strategy by paper scope

### Recommended no-cost primary study: fixed ETFs

Use 25--30 liquid US-listed ETFs, all trading before 2010, with yfinance daily
OHLCV. Create a membership table in which each selected ETF is eligible from
its pre-declared start date; then the current pipeline runs unchanged. This is
the only route that makes the public price source, investable universe, and
paper claim agree without pretending to own a historical stock-membership feed.

### Secondary open-data robustness: curated portfolio returns

Use free Kenneth French industry or international portfolio returns to validate
whether regime-aware allocation generalises beyond the ETF panel. This supports
graph/regime/allocation experiments based on return series, but not the
OHLCV-dependent DeepCNL graph.

### Exploratory stock replication: Kaggle / community membership data

It is acceptable for development and labelled exploratory results. It must be
presented as such unless membership, corporate actions, ticker mapping, and
delisting coverage have been independently audited. It should never be mixed
with the final ETF results in the paper's headline table.

### Long-horizon current-S&P survivor panel: Kaggle `darkmatternet`

The user-supplied Kaggle source is a valuable additional **development** panel:
its public metadata on 2026-07-26 reports version 118, 287,507,290 bytes,
CC0 licensing, and price rows for current S&P 500 companies from 2000 onward.
Its coverage makes it useful for training and stress-testing larger graph or
sequence models across multiple crises. It is not, however, a historical S&P
500 constituent dataset: the source explicitly defines its population as the
current companies, and therefore does not include firms that left the index.

When acquired, place the raw files in a separately versioned
`data/exploratory/` package with Kaggle version, retrieval timestamp, archive
hash, and an explicit `survivor_only_current_constituents` flag. Do not merge
its roster with the canonical point-in-time membership files or use it for an
index-replication/outperformance headline. It can still be a strong secondary
model-development and robustness resource, provided the manuscript labels it
accurately.

After manually downloading and unzipping `sp500_stocks.csv` from Kaggle, use
the strict importer below. It preserves the provider's `close` column rather
than assuming an adjustment convention, validates genuine OHLCV, hashes the
raw file, and writes a non-canonical survivor-panel sidecar. Use the exact
version shown by Kaggle at download time (118 was the inspected version).

```bash
python -B scripts/import_kaggle_current_sp500.py \
  --stocks /path/to/sp500_stocks.csv \
  --output data/exploratory/kaggle_current_sp500_v118/ohlcv.csv \
  --dataset-version 118 \
  --close-adjustment-convention 'provider_close_as_downloaded_verify_corporate_actions' \
  --drop-incomplete-dates \
  --drop-symbols-with-invalid-ohlcv
```

The importer intentionally does not generate a historical membership file.
That remaining data requirement cannot be recovered from a current-company
roster or its `date_added` column.

For the audited current-survivor development workflow, the cohort construction,
data-quality exclusions, experiment split, and paper-eligibility guardrail are
specified in [the Kaggle survivor exploratory runbook](KAGGLE_SURVIVOR_EXPLORATORY_RUNBOOK.md).
The initial baseline evidence is in the linked
[exploratory results note](KAGGLE_SURVIVOR_EXPLORATORY_RESULTS.md).

### Community Nifty historical-weight candidate

The downloaded version-4 Nifty source passes the narrow arithmetic and snapshot
checks: 216 dated 50-member snapshots from 2008-01 through 2025-08 and 100
historical symbols.  However, the matching public Yahoo audit retrieved usable
OHLCV for only 83 symbols, with 40--48 priced eligible members per decision
date and never the full 50.  It is therefore useful for an explicitly partial,
dynamic-universe exploratory study—not a Nifty 50 replication claim.  The
exact source hashes, exclusions, and build protocol are in the
[Nifty community exploratory runbook](NIFTY50_COMMUNITY_EXPLORATORY_RUNBOOK.md);
the development, frozen outer, and Transformer-feasibility results are in its
[exploratory results note](NIFTY50_COMMUNITY_EXPLORATORY_RESULTS.md).

The follow-up official NSE bhavcopy audit retrieved all 212 monthly archive
snapshots from 2008-01 through 2025-08 and finds 46--49 literal source-symbol
matches per 50-member snapshot (median 49). It confirms that official open
exchange files can materially improve coverage, but its prices are unadjusted
and several remaining gaps are ticker/identity transitions. The parser,
checksums, causality safeguard, and explicit non-backtest boundary are in the
[NSE bhavcopy coverage audit](NSE_BHAVCOPY_COVERAGE_AUDIT.md).

## Sources

- [Kaggle NYSE dataset description](https://www.kaggle.com/datasets/dgawlik/nyse)
- [Kaggle current-S&P 500 25-year OHLCV dataset](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily)
- [Kaggle NYSE mirror showing its documented coverage](https://baselight.app/u/kaggle/dataset/dgawlik_nyse)
- [yfinance source and usage notice](https://github.com/ranaroussi/yfinance)
- [FinRL's public-data recommendation](https://finrl.readthedocs.io/en/latest/faq.html)
- [Kaggle Nifty 50 historical-weights candidate](https://www.kaggle.com/datasets/megachonk/historical-nifty-50-constituent-weights-20y)
- [NSE capital-market historical reports](https://www.nseindia.com/resources/historical-reports-capital-market-daily-monthly-archives)
- [NSE UDiFF bhavcopy format transition](https://nsearchives.nseindia.com/content/circulars/CMTR62825.pdf)
- [S&P 500 historical-membership community repository](https://github.com/hanshof/sp500_constituents)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
