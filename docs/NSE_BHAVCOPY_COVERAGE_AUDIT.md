# Official NSE Bhavcopy Coverage Audit

## Purpose and boundary

This audit tests whether the public NSE equity-bhavcopy archive can recover the
historical price identities that Yahoo Finance no longer serves for the
community Nifty 50 membership prototype.  It is **not** a backtest dataset.
NSE bhavcopy close prices are raw exchange prices, so they cannot be appended
to the `auto_adjust=true` Yahoo panel or used for corporate-action-adjusted
return claims.

The reusable parser in `src/research/data/nse_bhavcopy_source.py` selects the
correct official format by date:

- Legacy archive through 2024-07-05:
  `archives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip`.
- UDiFF capital-market archive from 2024-07-08:
  `nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`.

Both schemas are parsed as ordinary `EQ` series with genuine exchange-reported
open, high, low, close, and volume.  Every sampled file URL and SHA-256 is
recorded in the output sidecar.  A browser-like request header is necessary
because the official archive otherwise rejects automated requests.

## Causal monthly probe, retrieved 2026-07-26

Using all 100 source ticker strings from the Kaggle Nifty membership candidate,
the coverage probe sampled the last available exchange day on or before each
calendar month-end from 2008-01 through 2025-08.  It successfully retrieved
all 212 requested months and produced 17,908 unadjusted observations covering
all 100 strings at least once.

Crucially, a weekend month-end is rolled **back**, never forward.  This avoids
the otherwise easy error where Pandas resolves a Sunday month-end to the next
Monday and leaks a post-decision price into an audit.  Exchange holidays are
handled by failed requests followed by explicitly recorded prior-day retries.

At the 216 community membership snapshots, literal source-symbol matching finds
46 to 49 of 50 eligible stocks (median 49): 1 snapshot has 46, 12 have 47, 52
have 48, and 151 have 49.  This is a material coverage improvement over the
Yahoo-only exploratory panel (40 to 48, median 45), but zero snapshots have a
literal full 50/50 match.

The remaining apparent gaps are mostly identifier transitions, not proof that
the exchange lacks the price history.  A dedicated 14-symbol alias probe
confirmed the following **observed exchange-code sequences** at monthly
snapshots.  These are coverage facts, not automatic return-history joins:

| Community ticker | Observed exchange-code sequence | Consequence for crosswalk |
| --- | --- | --- |
| `INFY` | `INFOSYSTCH` through 2011-05; `INFY` from 2011-06 | Candidate same-security name/symbol change. |
| `HEROMOTOCO` | `HEROHONDA` through 2011-07; `HEROMOTOCO` from 2011-08 | Candidate same-security name/symbol change. |
| `RELINFRA` | `REL` through 2008-04; `RELINFRA` from 2008-05 | Candidate same-security name/symbol change. |
| `BAJAJAUTO` | `BAJAJAUTO` through 2008-02; no March/April observation; `BAJAJ-AUTO` from 2008-05 | 2008 demerger/relisting boundary; separate pre/post assets and never splice raw returns. |
| `VEDL` | `SESAGOA` through 2013-09; `SSLT` from 2013-10 through 2015-04; `VEDL` from 2015-05 | Sesa Sterlite reorganisation plus later rename; only the post-reorganisation `SSLT`/`VEDL` name change can potentially be continuous after evidence review. |
| `VSNL` | `VSNL` in 2008-01; `TATACOMM` from 2008-02 | Historical restructuring/name transition requiring ISIN and event review. |
| `ETERNAL` | `ZOMATO` through 2025-03; `ETERNAL` from 2025-04 | Candidate recent name/symbol change. |

The exchange and issuer records independently support the need for this
classification: NSE records the 2008 Bajaj Auto demerger/relisting, and Vedanta
records describe the Sesa Goa/Sterlite reorganisation preceding the `SSLT` and
`VEDL` labels.  Those corporate changes are not comparable to a normal rename.

The observations above are **not** accepted mapping rules. A crosswalk must
retain source ticker, exchange ticker, ISIN, effective start/end, evidence URL,
and corporate-action/delisting treatment before it can become verified.

The legacy bhavcopy header (including the 2008 archive used here) has no ISIN
field; the newer UDiFF schema does report `ISIN`. The parser can retain that
newer identity field on request, but correctly refuses the same request for a
legacy archive rather than inventing a historical ISIN. Thus the legacy
2008–2024 identity links still require primary security-master/index-notice
evidence together with the action-ledger hints.

The probe also confirms that several histories Yahoo omitted are present in
the official exchange archive: `CAIRN` (111 sampled months), `HDFC` (186),
`IDFC` (201), `JPASSOCIAT` (194), `LTIM` (33), `MCDOWELL-N` (197),
`TATAMOTORS` (212), and `RELCAPITAL` (164).  This makes NSE bhavcopies a useful
open-source input to the eventual data build, but does not remove the
adjustment or identity requirements.

## Reproduce the audit

```bash
python -B scripts/probe_nse_bhavcopy_coverage.py \
  --symbols data/exploratory/kaggle_nifty50_weights_v4/yahoo_symbols_ns.csv \
  --symbol-column nifty_ticker \
  --start 2008-01-01 --end 2025-08-31 --pause-seconds 0.05 \
  --output data/exploratory/kaggle_nifty50_weights_v4/nse_bhavcopy_monthly_coverage_causal.csv
```

The output and `.source.json` sidecar are intentionally ignored exploratory
artifacts.  Their metadata marks them `raw_unadjusted_exchange_prices`, permits
historical-symbol coverage audit only, and explicitly forbids combining them
with the adjusted Yahoo panel or using them in paper backtests.

## Evidence-linked adjustment interface

`scripts/adjust_price_scale_actions.py` and
`src/research/data/corporate_actions.py` are ready for the next build stage.
They adjust only a reviewed split or bonus event with an explicit
`backward_price_factor`, source-record ID, and evidence URL. For a two-for-one
split, the factor is `0.5`: pre-event OHLC prices are halved and pre-event
volume is doubled. The event date and subsequent observations are unchanged.

The action input must contain `symbol`, `ex_date`, `action_type`,
`backward_price_factor`, `evidence_url`, and `source_record_id`. It refuses any
event type except `split_bonus_price_factor`; cash dividends, mergers,
spin-offs, rights issues, and delistings need separately specified value and
return treatment. Consequently, even a fully successful run creates a
**price-scale-adjusted**, not a total-return, series and does not pass the
paper-release gate by itself.

## Official corporate-action review ledger

The public NSE corporate-actions endpoint is now accessed through
`src/research/data/nse_corporate_actions_source.py`.  A bounded request has
the reproducible form:

`https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY`.

It retains only ordinary `EQ` records and preserves the exchange's raw action
description, ex date, record date, ISIN, company name, face value, closure
dates, source URL, and a deterministic source-record ID.  A 2024-01
whole-equity schema check returned 43 records, including `ALLCARGO` with the
literal subject `Bonus 3:1`; it did **not** create a `0.25` factor or touch a
price series. That distinction prevents a free-text action description from
silently becoming an accounting convention.

Run a strict, monthly-bounded review collection with:

```bash
python -B scripts/probe_nse_corporate_actions.py \
  --start 2008-01-01 --end 2025-08-31 \
  --symbols-csv data/exploratory/kaggle_nifty50_weights_v4/membership_snapshots.csv \
  --output data/exploratory/kaggle_nifty50_weights_v4/nse_corporate_actions_review.csv
```

The script refuses to write a partial ledger if any month fails, records each
raw response hash in the sidecar, and marks the output
`unreviewed_events_not_price_adjustments`. The endpoint may repeat an exact
record in adjacent query windows; only byte-equivalent normalized records are
collapsed, with the count recorded in the sidecar. Any conflicting duplicate
halts collection. A reviewer must decide whether a specific event is a
supported split/bonus price factor, a total-return cash event, or an
identity/value transition before the adjustment or crosswalk tools can use it.

For longer periods, first collect complete non-overlapping batches and then
use `scripts/merge_nse_corporate_action_ledgers.py`. It validates every input
sidecar and date window, preserves all input and sidecar hashes in the merged
manifest, and reruns the same exact-replay reconciliation across batch
boundaries. The 2008-01 through 2025-08 collection currently consists of
seven complete batches and 34,425 review-only records after this merge. This
is an action-evidence ledger—not an adjusted-price or total-return dataset.

### Split/bonus review queue

`scripts/screen_nse_price_scale_actions.py` classifies conventional split and
bonus wording into a **manual** review queue. For the 100 community source
symbols, it found 107 split/bonus-like descriptions: 102 conventional factor
proposals requiring human approval and 5 blocked complex events (debentures,
preference shares, or schemes). It does not generate the
`action_type/backward_price_factor` columns accepted by the adjustment runner.

```bash
python -B scripts/screen_nse_price_scale_actions.py \
  --action-ledger data/exploratory/kaggle_nifty50_weights_v4/nse_corporate_actions_review_2008_2025_08.csv \
  --symbols-csv data/exploratory/kaggle_nifty50_weights_v4/membership_snapshots.csv \
  --output data/exploratory/kaggle_nifty50_weights_v4/nse_price_scale_action_review_community_symbols.csv
```

A proposal such as `Bonus 1:1` or a face-value split is not approval to change
history. Each approved row must be independently checked against its linked
record and then explicitly entered into the guarded adjustment input.

## Time-bounded identity contract

`scripts/apply_identity_crosswalk.py` joins raw provider prices and the
membership history through a reviewed crosswalk rather than assuming that a
ticker string is a permanent security. Each crosswalk row requires:

`source_symbol`, `asset_id`, `provider_symbol`, effective start/end dates,
ISIN, relationship type, continuity classification, verification status,
evidence URL, and source-record ID.

The only relationship allowed to preserve a rolling return history is
`same_security_rename` with `continuity=continuous` and
`verification_status=verified`. Any predecessor/successor reorganisation must
be declared with `continuity=break` and a different `asset_id`. The runner
refuses candidate or blocked rows, overlapping dates, and unmapped records by
default. This makes it impossible to accidentally turn a corporate demerger
into a harmless-looking symbol rename.

The initial audit has already identified why this matters.  Exchange and
issuer records show historic code changes such as `INFOSYSTCH`/`INFY` and
`REL`/`RELINFRA`, which may become verified continuous mappings after their
effective dates and ISIN evidence are captured. By contrast, the 2008 Bajaj
Auto demerger and the Sesa Goa/Vedanta combination require value and
successor/predecessor treatment, not a direct price splice. No such candidate
is used until its evidence record is reviewed.

### Action-ledger ISIN screening

`scripts/screen_nse_action_identity_hints.py` turns the merged action ledger
into an explicitly non-authoritative review queue. Against the 100 community
source ticker strings, 91 have one candidate ISIN in the action ledger and 9
have no action-ledger observation: `BAJAJAUTO`, `ETERNAL`, `IBULHSGFIN`,
`INFRATEL`, `LTIM`, `MCDOWELL-N`, `SESAGOA`, `TATAMOTORS`, and `VSNL`.

This is useful triage, not a crosswalk: the NSE action API can retrospectively
emit a later/current code for an older event. For example, its historical
events use `INFY` and `HEROMOTOCO` where the raw bhavcopy sequence shows
`INFOSYSTCH` and `HEROHONDA`. Consequently all 91 ISIN rows are labelled
`candidate_not_verified`; they cannot be passed to the identity-crosswalk
runner until exact effective dates, provider codes, continuity, and primary
evidence are reviewed.

```bash
python -B scripts/screen_nse_action_identity_hints.py \
  --action-ledger data/exploratory/kaggle_nifty50_weights_v4/nse_corporate_actions_review_2008_2025_08.csv \
  --symbols data/exploratory/kaggle_nifty50_weights_v4/membership_snapshots.csv \
  --output data/exploratory/kaggle_nifty50_weights_v4/nse_action_identity_hints_community_symbols.csv
```

## Required next data work

1. Reconcile every community membership ticker to time-bounded NSE ticker and
   ISIN identities using official Nifty index notices/security masters.
2. Review the newly collectable official action ledger and apply a fully
   auditable split, bonus, rights, dividend, merger, and delisting-return
   policy consistently to all OHLC fields.
3. Download daily—not merely monthly—raw bhavcopies only after the identity and
   adjustment design has been frozen; preserve raw files/hashes and retry logs.
4. Rebuild the validation and all baselines from that single adjusted panel;
   do not splice isolated raw observations into the existing Yahoo experiment.

Until these steps are independently checked, this remains an exploratory
coverage finding and all India results remain excluded by the paper-release
gate.
