# NSE Historical Identity Evidence Audit

## Outcome

The versioned evidence ledger
[`nse_identity_evidence_v1.csv`](../configs/nse_identity_evidence_v1.csv) is
complete for the 13 records in
[`nse_identity_crosswalk_candidates_v1.csv`](../configs/nse_identity_crosswalk_candidates_v1.csv),
but **zero records are eligible for promotion to `verified`**. This is an
intentional fail-closed result: the candidate crosswalk must not be used to
construct a continuous Indian equity return panel.

Run the audit with:

```bash
python -B scripts/audit_identity_evidence.py \
  --crosswalk configs/nse_identity_crosswalk_candidates_v1.csv \
  --evidence configs/nse_identity_evidence_v1.csv
```

The audit attaches an evidence row to every mapping and enforces the following
minimum requirement before a mapping can be marked `verified`:

1. a source establishes the historical provider symbol;
2. a source establishes the complete effective interval;
3. a source establishes the ISIN; and
4. a source establishes uninterrupted economic-security continuity.

At least one supporting source must be an official exchange or bhavcopy source.
This checks identity only. It does not establish total-return adjustment,
delisting treatment, point-in-time membership, or paper eligibility.

## What the primary sources establish

| Candidate | Strongest evidence now available | Reason it stays a candidate |
| --- | --- | --- |
| `ZOMATO` → `ETERNAL` | NSE Circular 54/2025 specifies both symbols and the 2025-04-09 effective date. | The circular does not give an ISIN. |
| `REL` → `RELINFRA` | Reliance Infrastructure lists both NSE codes and ISIN `INE036A01016`. | It does not supply the historical effective interval. |
| `HEROHONDA` → `HEROMOTOCO` | Hero's annual report records the 2011-07-28 name-change certificate; bhavcopies show the observed code switch. | No dated historical exchange master ties both codes to the ISIN. |
| `INFOSYSTCH` → `INFY` | Issuer material and a historical NSE bhavcopy support the legacy code and observed switch. | The legacy bhavcopy format does not provide an ISIN or continuity proof. |
| `VSNL` → `TATACOMM` | Tata Communications describes itself as the new name of VSNL; bhavcopies show the later code switch. | No dated exchange-security/ISIN link. |
| `SSLT` → `VEDL` | The issuer's release describes Sesa Sterlite in the context of an amalgamation/reorganisation. | This is not evidence of an uninterrupted pre-reorganisation return history, nor a dated symbol–ISIN link. |
| Bajaj Auto relisting | NSE describes the demerger/relisting boundary. | It is a continuity break, never a predecessor-price splice. |

The source links and precise, intentionally narrow claims are stored in the
CSV rather than inferred from company names. The Vedanta source was visually
checked from the original PDF: it describes an amalgamation/reorganisation, so
the project does not treat it as a simple rename.

## Gate to build the India confirmation panel

Before running any India backtest, obtain a dated NSE historical security
master or equivalent exchange record that links each provider symbol to its
ISIN over its valid interval. Then separately establish all of the following:

1. a point-in-time constituent-membership history;
2. a policy and evidence for splits, bonuses, rights, mergers, demergers,
   delistings, and cash distributions; and
3. adjusted/total-return prices consistent with that policy.

Only a fully verified crosswalk plus those data contracts may turn the current
official-NSE exploration into a paper-confirmation market. Until then, the
India material remains an audit trail and a data-acquisition target, not a
reported ICAIF result.
