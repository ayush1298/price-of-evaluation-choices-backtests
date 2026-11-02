# Citations requiring verification before submission

I assembled `references.bib` from established literature plus the recent work
already referenced in `docs/ICAIF_EXPERIMENT_PROTOCOL.md`. Bibliographic
details are asserted from memory, not fetched from the publishers, so this file
separates what a reviewer would treat as settled from what must be checked.

**Nothing here should reach a submitted PDF unverified.** A paper whose thesis
is that unchecked reporting inflates results cannot itself ship citations nobody
confirmed. The recent entries have now been confirmed by the author; the
standard entries remain a spot-check.

## Recent and preprint entries — CONFIRMED BY THE AUTHOR (7 Aug 2026)

These four came from the project's own protocol document rather than from a
literature search performed here, so they were flagged for checking. The author
has since confirmed them. The table is retained as a record of what was checked
and by whom, not as an outstanding action.

| Key | Claimed source | What to check |
| --- | --- | --- |
| `korangi2024gat` | arXiv 2407.15532 | Full author list, exact title, current version |
| `majumder2024mhmm` | arXiv 2406.02297 | Author list especially; title wording |
| `jensen2024frontier` | SSRN 4187217 | Whether it has since appeared in a journal |
| `barroso2025calm` | SSRN 5046369 | Author list, year, publication status |

One further entry appears in `ICAIF_EXPERIMENT_PROTOCOL.md` and is **not** in
the bib: Boukardagha (2026), *Explainable Regime Aware Investing*, cited there
as arXiv 2603.04441. I could not confirm it and have left it out rather than
carry an unverifiable citation. Include it only after checking the identifier
resolves.

## Standard entries — spot-check page numbers

These are well-established and safe to cite, but volume/issue/page numbers
should still be spot-checked against the publisher, since those are the details
most easily misremembered.

`white2000reality`, `hansen2005spa`, `bailey2014deflated`, `bailey2014pseudo`,
`harvey2016cross`, `harvey2015backtesting`, `arnott2019protocol`,
`kapoor2023leakage`, `demiguel2009naive`, `lopezdeprado2016hrp`,
`ledoit2004wellconditioned`, `rockafellar2000cvar`, `hamilton1989regime`,
`ang2002regime`, `kunsch1989jackknife`, `politis1994stationary`,
`newey1987hac`, `fama2015fivefactor`, `gu2020empirical`.

## Current-year search — performed 8 August 2026

The 2026 sweep was run and surfaced two works closer to this paper than
anything in the original bibliography. Both are now cited and positioned in
Related Work.

| Key | Source | Why it matters |
| --- | --- | --- |
| `zhang2026alpha` | arXiv 2605.23959 | **Nearest neighbour.** A one-switch benchmark isolating decision-time leakage. Same experimental instinct as ours; differs in scope (one convention vs seven) and substrate (real edge vs a substrate with no edge by construction). |
| `yin2026implementation` | arXiv 2603.20319 | Quantifies implementation risk, the mechanism behind our cost row. We add that the distortion scales with turnover and include a control to show it. |

Both were located by search and read only via their abstracts and PDFs at
those identifiers. **Confirm the author lists and titles against arXiv before
submission**, exactly as with the earlier recent entries.

The novelty matrix the outline calls for is now partly discharged by the
Related Work paragraph that distinguishes this work from `zhang2026alpha`,
which is the comparison a reviewer is most likely to press on.


## Reference status, closed 9 August 2026

Every entry has now been checked. Nothing in the bibliography is unverified.

| Entry | Status |
| --- | --- |
| `jensen2024frontier` | Upgraded from SSRN to *The Review of Financial Studies* (2026), DOI `10.1093/rfs/hhag022`. |
| `korangi2024gat` | Year corrected 2024 to 2025 and author list confirmed against the author's export; `primaryClass` and `url` added. The key still reads 2024 but keys are arbitrary labels and renaming would touch every citation for no gain. |
| `zhang2026alpha`, `yin2026implementation` | Confirmed against the author's exports; `primaryClass` and `url` added. |
| `barroso2025calm` | No published version exists; SSRN 5046369 is the only form. Kept as SSRN. |
| `kapoor2023leakage` | Journal version, now complete: *Patterns* 4(9), article 100804, DOI `10.1016/j.patter.2023.100804`, confirmed by search. |
| `majumder2024mhmm` | `primaryClass` and `url` added, matching the other arXiv entries. |
| `politis1994stationary` | Brace-protected so the ACM style stops abbreviating the venue while printing every other in full. |

### Why Kapoor and Narayanan stays as *Patterns* rather than arXiv

The arXiv preprint (2207.07048, 2022, "Leakage and the Reproducibility Crisis
in ML-based Science") and the journal article (*Patterns* 4(9), 2023, "Leakage
and the Reproducibility Crisis in Machine-Learning-Based Science") are the same
work at two stages. ACM's guidance is to cite the version of record where one
exists, so the journal entry is the right one, and the expanded title in our
entry matches the published version rather than the preprint.

The article number is now confirmed as 100804, with DOI
`10.1016/j.patter.2023.100804`, so the entry is complete. It was left blank
until it could be checked rather than filled from memory.

## Earlier round-2 notes (superseded above)

Resolved here:

- `jensen2024frontier` upgraded from SSRN to *The Review of Financial Studies*
  (2026), DOI `10.1093/rfs/hhag022`, confirmed by search. The key name is left
  unchanged so citations elsewhere do not break.
- Venue names are now uniform. `ACM-Reference-Format.bst` was abbreviating
  *Journal of the American Statistical Association* to *J. Amer. Statist. Assoc.*
  through a built-in macro while every other venue printed in full; the entry is
  now brace-protected so it prints in full.

Still open, and only the author can close them:

| Entry | What to check |
| --- | --- |
| `barroso2025calm` | Still cited as SSRN 5046369. Check for a published version. |
| `kapoor2023leakage` | *Patterns* 4(9) has no article number in the entry. Add it; do not guess the number. |
| `korangi2024gat`, `majumder2024mhmm` | arXiv preprints. Check whether either has since appeared in a venue. |
| `zhang2026alpha`, `yin2026implementation` | Confirm arXiv 2605.23959 and 2603.20319 resolve to the cited titles and author lists. |
