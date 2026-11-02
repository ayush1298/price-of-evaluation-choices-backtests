# Validation Candidate Ledger and Frozen Selection

The candidate ledger implements the readiness-plan rule that an outer-test
configuration cannot be changed using its own results. Before a candidate
validation cycle begins, commit a ledger derived from
[`validation_candidate_ledger_TEMPLATE.json`](../configs/ledgers/validation_candidate_ledger_TEMPLATE.json).
It names every allowed candidate, validation window, data contract, selection
metric, canonical run ID, and expected manifest values.

After every validation run finishes, audit the ledger:

```bash
python -B scripts/audit_validation_candidate_ledger.py \
  --ledger configs/ledgers/your_frozen_ledger.json \
  --results-root results/canonical
```

Selection is ready only when every candidate has a manifest and metrics file
that match the committed ledger. Then—and only then—freeze the winner before
opening the outer period:

```bash
python -B scripts/freeze_validation_selection.py \
  --ledger configs/ledgers/your_frozen_ledger.json \
  --results-root results/canonical \
  --output configs/selections/your_frozen_selection.json
```

The frozen record includes the selected validation-run config hash and expected
runner values. It must be recorded as an input to the outer-test run. A
missing, mismatched, or post-hoc candidate blocks selection instead of quietly
dropping it.

## Historical HMM audit

[`public_us_etf_hmm_uncertainty_archival_v1.json`](../configs/ledgers/public_us_etf_hmm_uncertainty_archival_v1.json)
captures the earlier 2/3/4-state by 63/126/252-session development grid solely
for traceability. It is explicitly `selection_permitted=false`: the results
were already observed before this ledger existed and cannot be retroactively
used to authorise a confirmation run.

The first audit found eight matching run artefacts and one absent one:
`public-etf-dev-hmm-risk-h3-l252-b0-v1`. That run was regenerated on
6 August 2026 from its unchanged frozen config, contract, and inputs, and
reproduced the documented metrics exactly (Sharpe 0.9228, 6.71% annualised,
-12.46% drawdown, 1,489 observations) at a later code revision. The audit now
reports 9/9 complete.

Two distinct properties must not be conflated:

- the grid is now **auditable**, so multiplicity-aware inference can run over
  the complete search family (see
  [`MULTIPLICITY_AWARE_INFERENCE.md`](MULTIPLICITY_AWARE_INFERENCE.md), White
  Reality Check p=0.680); and
- the grid is still **not selectable**. `selection_permitted` remains false
  because the results were observed before the ledger existed. Completing an
  artefact set retrospectively cannot convert observed evidence into a valid
  precommitment.

The HMM candidate remains an archived negative ablation; no outer or
confirmation run may be launched from it.
