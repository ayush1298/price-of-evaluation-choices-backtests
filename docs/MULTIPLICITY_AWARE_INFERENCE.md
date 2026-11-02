# Multiplicity-Aware Inference

Paired bootstrap compares one preselected strategy with one benchmark. It does
not correct for trying several HMM states, lookbacks, graph settings, or neural
variants and reporting the best. The canonical pipeline now also provides a
moving-block implementation of **White's Reality Check** over every candidate
named in a versioned candidate ledger.

```bash
python -B scripts/white_reality_check_canonical_runs.py \
  --candidate-ledger configs/ledgers/your_validation_ledger.json \
  --benchmark-run results/canonical/your_matched_baseline \
  --results-root results/canonical \
  --output results/your_reality_check.json
```

The command refuses an incomplete ledger and refuses candidates whose contract,
walk-forward window, or backtest settings differ from the benchmark. It
resamples common blocks of all candidate-minus-benchmark net-return
differentials, so each bootstrap draw preserves both time dependence and the
cross-candidate relation induced by the same market day.

Use this alongside the existing paired moving-block comparison and deflated
Sharpe diagnostic. A Reality Check p-value is not a substitute for an
untouched outer test, total-return/point-in-time data validation, or a second
market confirmation; it addresses only model-search multiplicity.

## Completed HMM search-family result

The historical HMM ledger initially could not be tested because one of its nine
expected run artefacts was absent. That artefact,
`public-etf-dev-hmm-risk-h3-l252-b0-v1`, was regenerated on 6 August 2026 from
its unchanged frozen config and inputs; it reproduced the documented Sharpe
0.9228, annualised return 6.71%, drawdown -12.46%, and 1,489 observations at a
later code revision. The ledger now audits 9/9 complete.

The matched benchmark is the 756-session unconditional HRP control. It was
regenerated as `public-etf-dev-hrp-pcc-lookback756-control-v2` so that it
carries the same data-contract provenance and comparable context as the
candidates; its net daily returns are bit-identical to the original control run
(maximum absolute difference 0.0, Sharpe 0.956389).

```bash
python -B scripts/white_reality_check_canonical_runs.py \
  --candidate-ledger configs/ledgers/public_us_etf_hmm_uncertainty_archival_v1.json \
  --benchmark-run results/canonical/public-etf-dev-hrp-pcc-lookback756-control-v2 \
  --results-root results/canonical \
  --block-length 21 --replications 5000 --seed 7 \
  --output results/public_etf_hmm_ledger_reality_check.json
```

| Quantity | Value |
| --- | ---: |
| Candidates in family | 9 |
| Paired daily observations | 1,489 |
| Best candidate | `hmm-2-lookback-252-band-0` |
| Best annualised excess return over control | 0.111% |
| **White Reality Check p-value** | **0.680** |

This is the quantity the single-candidate test cannot supply. The naive paired
bootstrap for the same best cell against the same control gives p=0.359; once
the other eight searched configurations are admitted, the family-wise p-value
rises to 0.680. The apparent development maximum is therefore consistent with
search noise, which is the correct reading of a nine-cell grid whose best cell
exceeds its control by 0.005 Sharpe.

Restoring the missing artefact makes the family *auditable*; it does not make
the ledger selectable. `selection_permitted` remains false, the results were
observed before the ledger existed, and no confirmation run may be launched
from it.
