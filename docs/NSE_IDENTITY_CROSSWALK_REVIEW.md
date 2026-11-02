# NSE Historical Identity Crosswalk Review

## Status

[`nse_identity_crosswalk_candidates_v1.csv`](../configs/nse_identity_crosswalk_candidates_v1.csv)
is a review queue, not an executable identity map. Every row is deliberately
`verification_status=candidate`; `apply_identity_crosswalk.py` will reject it
for a paper or adjusted-panel build. The table records the best currently
available primary issuer/exchange evidence together with exact observed
bhavcopy code-switch dates.

The linked [`nse_identity_evidence_v1.csv`](../configs/nse_identity_evidence_v1.csv)
records which facts each source actually establishes. Its reproducible audit
currently finds zero fully supported mappings; see
[`NSE_IDENTITY_EVIDENCE_AUDIT.md`](NSE_IDENTITY_EVIDENCE_AUDIT.md). A plausible
company-name match or a current ISIN is therefore not enough to promote a row.

This distinction matters: a ticker can be a simple renamed security, a code
change following a reorganisation, or a source-data label that appeared only
after the economic event. The table makes the evidence and remaining gap
visible rather than implicitly treating every alias as continuous.

## Confirmed observations, awaiting crosswalk verification

| Source label | Raw provider sequence | Observed switch | Evidence classification |
| --- | --- | --- | --- |
| `INFY` | `INFOSYSTCH` → `INFY` | 2011-06-29 | Infosys' 2011 report identifies `INFOSYSTCH` as its former NSE code. The final decision still needs a direct exchange name/symbol notice or historical security-master record. |
| `HEROMOTOCO` | `HEROHONDA` → `HEROMOTOCO` | 2011-08-08 | Hero's annual report records the same company's name-change certificate dated 2011-07-28. The ISIN is a strong candidate, but the review must retain the exchange-code evidence and confirm no share-class change. |
| `RELINFRA` | `REL` → `RELINFRA` | 2008-05-16 | Reliance Infrastructure's investor page identifies `REL` as physical and `RELINFRA` as electronic NSE codes for ISIN `INE036A01016`. The historical effective-date record is retained in the probe sidecar. |
| `TATACOMM` | `VSNL` → `TATACOMM` | 2008-02-25 | Tata Communications states it is the new name of VSNL as of 2008-01-28. The later trading-code date must remain explicit. |
| `VEDL` | `SSLT` → `VEDL` | 2015-05-07 | The issuer documents Sesa Sterlite's later name change to Vedanta. This row does **not** splice pre-2013 Sesa Goa/Sterlite reorganisation history into the asset. |
| `ETERNAL` | `ZOMATO` → `ETERNAL` | 2025-04-09 | NSE Circular 54/2025 directly specifies the old/new names and symbols effective this date. It does not itself provide the ISIN needed for a promotable historical identity record. |
| `BAJAJAUTO` | `BAJAJ-AUTO` post-2010 source mapping | n/a | NSE records a 2008 demerger and relisting. It is classified `reorganisation_successor` with `continuity=break`, never as a return-history splice. |

## What remains before promotion to `verified`

1. Retrieve a primary exchange notice/security-master record that establishes
   the provider code and ISIN over each complete interval, particularly for
   legacy bhavcopies that do not publish ISIN.
2. Confirm whether the original community membership label was economically
   valid over the proposed interval; do not repair a bad source label merely
   because a plausible successor exists.
3. Record treatment for every merger, demerger, delisting, rights issue, and
   cash distribution. A simple-name mapping can be promoted only when those
   events do not break the security's return history.
4. Add all unresolved symbols, then require a fully `verified`, complete
   crosswalk before using the official NSE panel in a paper backtest.

The raw daily probe files and their sidecars are ignored exploratory artifacts.
They preserve archive URL and SHA-256 evidence for the dates above; the CSV
candidate table is intentionally versioned so review decisions are visible.
