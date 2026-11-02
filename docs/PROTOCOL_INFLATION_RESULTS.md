# Protocol-Inflation Results

## What this measures

Every regime-aware, graph, and neural candidate in this project failed its
held-out gate.  That makes the evaluation *protocol* the measurable object
rather than the models: given the same manifest-backed runs, how much higher
does a reported number look when it is read under a permissive convention
instead of the strict one?

Each row below holds the panel, strategy, allocator, cost model, execution lag,
and cap fixed, and varies only the reporting choice.  Nothing is refit,
repriced, or selected.  The rows are fixed in
[`protocol_inflation_table_v1.json`](../configs/protocol_inflation_table_v1.json)
so the row set cannot be grown after seeing which rows look most striking.

```bash
python -B scripts/build_protocol_inflation_table.py \
  --spec configs/protocol_inflation_table_v1.json \
  --results-root results/canonical \
  --output results/protocol_inflation_table_v1.json \
  --markdown-output docs/tables/protocol_inflation_v1.md
```

The generator refuses to reduce runs that cover different evaluation lengths and
refuses to silently drop a named run, so a row cannot be quietly narrowed to the
runs that favour it.  Every input run's manifest and metrics hash is recorded in
the output JSON.

## Results

| Evaluation shortcut | Panel / strategy | Reported | Honest | Inflation |
| --- | --- | ---: | ---: | ---: |
| Development window reported as held-out | US 30-ETF: dynamic graph-temporal, 5 epochs | 0.775 (5-seed mean) | 0.685 (5-seed mean) | **+0.090** |
| Best random seed instead of the seed distribution | US 30-ETF: dynamic graph-temporal, outer | 0.734 (max) | 0.729 (median) | **+0.005** |
| Best random seed instead of the seed distribution | US 50-ETF: dynamic graph-temporal, development | 1.261 (max) | 0.759 (median) | **+0.502** |
| Gross / near-zero transaction costs | US 30-ETF: dynamic graph-temporal (56%/month turnover) | 0.731 (0 bps) | 0.500 (50 bps) | **+0.232** |
| Gross / near-zero transaction costs | US 30-ETF: **equal weight control** (2.6%/month turnover) | 0.720 (0 bps) | 0.710 (50 bps) | **+0.010** |
| Best grid cell without correcting for the search | US 30-ETF: HMM conditional risk, 9-cell grid vs matched HRP-756 | p=0.359 | p=0.680 | **+0.321** |
| Gross / near-zero costs | **Track-B sealed**: dynamic graph-temporal seed 7 (48.9%/month) | 0.735 (0 bps) | 0.568 (50 bps) | **+0.167** |
| Gross / near-zero costs | **Track-B sealed**: equal weight control (2.0%/month) | 0.680 (0 bps) | 0.673 (50 bps) | **+0.007** |
| **Fix one favourite baseline instead of a ladder** | **Track-B sealed**: best seed vs six pre-declared baselines | **+0.045 edge** | **-0.025 edge** | **+0.070, sign flips** |
| Best candidate without correcting for the family | **Track-B sealed**: 7-candidate family vs strongest baseline | p=0.131 | p=0.344 | **+0.213** |

All allocation rows are net Sharpe.  Inflation is reported minus honest.  Rows
marked **Track-B sealed** were computed on the confirmation panel, which had
never been examined when the earlier rows were produced.

## Reading the rows

**The cost row contains its own control.**  The same shortcut that adds 0.232
Sharpe to a 56%-per-month-turnover neural model adds 0.010 to equal weight.
The convention is not neutral: it transfers apparent skill to whichever method
trades most.  A paper that reports gross returns is therefore not merely
optimistic, it is optimistic *in the direction of its own proposed method*.

**Seed instability is panel-dependent, not universal.**  On the 30-ETF outer
window the best-versus-median seed gap is 0.005 and best-seed reporting would
barely mislead.  On the 50-ETF development panel the same model, allocator, and
data give a 1.079 Sharpe spread across the five prescribed seeds (1.261 down to
0.182, median 0.759).  A single-seed result is not evidence, and the number of
seeds required to notice that is a property of the panel rather than a constant
a reader can assume.

**Multiplicity dominates the inference.**  The nine-cell HMM grid's best cell
beat its matched control by 0.005 Sharpe with a single-comparison p of 0.359.
Admitting the eight other configurations that were searched moves the
family-wise p to 0.680.  Neither supports the candidate; the point is that the
uncorrected number is roughly half the corrected one, and only the corrected one
answers the question the search actually asked.

**Benchmark choice can flip the conclusion outright.**  This is the strongest
row in the table and the only one that changes the *sign* of a result rather
than its magnitude.  Equal weight was the strongest simple comparator on the
development panel, so it is the baseline a study following its own development
evidence would carry forward.  Against it, the best confirmation seed shows a
+0.045 Sharpe edge — a positive result.  Against the strongest baseline actually
present on that panel, the same run shows -0.025 — a negative result.  Nothing
about the model, data, window, or cost model differs between those two
sentences; only the comparator does.  Reporting a baseline ladder rather than a
baseline is therefore not a courtesy, it is what makes the claim decidable.

**The cost result replicates out of sample.**  The +0.232 / +0.010 contrast was
measured on the development panels.  Repeated on the sealed confirmation panel
it gives +0.167 for the 48.9%-turnover model against +0.007 for the
2.0%-turnover control.  The inflation *measurement itself* therefore replicates
on data that was untouched when the measurement was designed, which is a
stronger claim than any single panel can support.

**Development-window reporting is the mildest shortcut here** at +0.090, which
is worth stating plainly: it is the shortcut this literature most often
discusses, and on this evidence it is not the most damaging one.  Cost
assumptions and seed reporting matter more.

## Training budget

The neural comparator was frozen at five epochs while its configuration file
declared twenty, which is the obvious objection to the whole study: perhaps
nothing works here only because the comparator was undertrained. Running the
declared budget on the development panel, five seeds:

| Window | 5 epochs | 20 epochs | Equal weight |
| --- | ---: | ---: | ---: |
| Development (2014--2019) | 0.763 | **0.930** | 0.792 |
| Held out (2020+) | 0.729 | **0.604** | 0.718 |

At twenty epochs the model beats equal weight on the development window by
$+0.137$ and loses to it out of sample by $-0.115$. The extra training buys
0.167 Sharpe where it is measured and returns none of it where it is tested, a
0.326 swing between the two readings. Seed spread widens from 0.185 to 0.321.

This is the objection answered rather than avoided, and the answer is the
paper's own thesis on an axis we had not measured: training the comparator
properly improves its development number and leaves its honest number no better.
The five-epoch configuration is the one reported throughout because it is what
was frozen and carried to confirmation. The twenty-epoch model was **not** run
on the sealed panel; the pre-registration permits one cycle, and evaluating a
new configuration there would be exactly the search this paper measures.

## What this table is not

It is not evidence that any method works.  Every candidate involved failed its
held-out gate, and the honest column of the best row is still below the equal
weight comparator over the same window (0.718).  The table measures reporting
conventions.  It does not license a performance claim, and no row may be quoted
as a result for the underlying strategy.

The magnitudes are specific to these panels, these strategies, and these
windows.  They are a demonstration that the inflation is measurable and
material, not a general correction factor to subtract from other people's
published Sharpe ratios.

## Provenance

- Generated table: `results/protocol_inflation_table_v1.json` (records the
  SHA-256 of every run manifest and metrics file it read).
- Reality Check input: `results/public_etf_hmm_ledger_reality_check.json`, see
  [`MULTIPLICITY_AWARE_INFERENCE.md`](MULTIPLICITY_AWARE_INFERENCE.md).
- Underlying run evidence: [`PUBLIC_ETF_EXPERIMENT_RUNBOOK.md`](PUBLIC_ETF_EXPERIMENT_RUNBOOK.md),
  [`EXPANDED_US_ETF_EXPERIMENT_RUNBOOK.md`](EXPANDED_US_ETF_EXPERIMENT_RUNBOOK.md),
  and [`HMM_UNCERTAINTY_REGIME_RISK_RESULTS.md`](HMM_UNCERTAINTY_REGIME_RISK_RESULTS.md).
