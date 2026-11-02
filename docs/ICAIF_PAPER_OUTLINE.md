# ICAIF '26 Paper Outline

**Target:** ACM ICAIF '26, Milan. Extended deadline 9 August 2026. Eight pages
in ACM `sigconf` two-column format including figures and references.
Double-blind: third-person self-citation, anonymised artefact link.

**Working title:** *How Much of Regime-Aware and Graph-Neural Portfolio
Outperformance Is an Evaluation Artefact? A Pre-Registered, Cost-Aware Audit*

**Claim being made:** that common evaluation conventions inflate reported
portfolio results by measurable, material amounts, demonstrated on an audited
benchmark of manifest-backed runs, with a pre-registered confirmation on a
held-out panel.

**Claim explicitly not being made:** that any regime-aware, graph, or neural
method here outperforms. None does. That is the premise, not a failure to
disclose.

## Section plan (8 pages)

### 1. Introduction (~0.75 p)

The literature reports strong regime-aware and graph-neural portfolio results.
This work builds the same family of methods under one causal, cost-aware
pipeline, finds none of them beats a simple baseline, and then uses that
negative substrate to measure what each evaluation shortcut would have bought.

Contributions:

1. an audited benchmark of manifest-backed causal runs across two markets and
   three fixed-instrument panels, under one shared allocator and cost model;
2. a **protocol-inflation measurement**: the Sharpe (and p-value) cost of five
   common reporting conventions, each isolated by holding everything else fixed;
3. a **pre-registered confirmation** on a panel with zero instrument overlap,
   where the prediction of failure was committed before the panel was opened;
4. a release gate and reproducibility harness that fails closed, with every
   number traceable to a run manifest hash.

### 2. Related work (~0.75 p)

Graph-attention portfolio optimisation; multivariate HMM regime allocation; the
implementable efficient frontier; forecast/weight stabilisation; explainable
regime-aware investing. Position: this paper does not propose a competitor to
these. It asks what an evaluation protocol has to look like before any of their
comparisons is decidable. Cite the backtest-overfitting and deflated-Sharpe
literature as the closest prior work, and state the difference — those bound
*how much* a Sharpe should be discounted in principle; this measures what
specific conventions cost on a fixed, re-runnable substrate.

**Action:** fresh literature search for the submission year before drafting;
one-page novelty matrix (paper / data / regime construction / allocator /
evaluation / precise difference). Add a method only if it changes a
pre-registered hypothesis.

### 3. Experimental apparatus (~1.5 p)

- Locked decision timeline: at month-end *t*, only data through *t*; no scaler,
  graph edge, regime feature, hyperparameter, membership, or label may use a
  later observation.
- One shared allocator and backtester for every method, so a learned score
  cannot win by receiving looser constraints.
- Data contracts: declared universe, adjustment convention, source sidecars,
  coverage gates, and hashes. Contract validation is a precondition of a run.
- Manifest-backed runs: config, input hashes, code revision, targets,
  executions, holdings, daily NAV, diagnostics, seeds.
- The mechanical release gate: `paper_eligible` false by default; five
  independent evidence flags required; the summariser refuses to export an
  unclassified run.

Figure 1: the decision timeline and the leakage points each control closes.

### 4. The methods and why none of them wins (~1.5 p)

Compressed results ladder — baselines, graph/selection variants, regime
routing, and the neural comparators — reported as one table with the
development gate outcome for each. Emphasise the matched-pair discipline: every
comparison is on identical dates via paired moving-block bootstrap.

Key negative findings, stated once and without hedging:

- soft-regime conditional covariance: p=0.909 vs matched control;
- HMM uncertainty-aware risk: best of nine cells, p=0.359 uncorrected;
- group-aware conditional covariance: p=0.732;
- graph attention vs similarity propagation: p=0.474;
- IC objective: US inconclusive, India p=0.998 against;
- gated regime experts: worse than regime-free;
- no positive five-factor alpha for any strategy on the public panels.

The one internally positive result must also be reported: the full
graph-temporal model beats its own graph-only (p=0.007) and temporal-only
(p=0.023) ablations, while failing to beat equal weight (p=0.705). This is the
paper's cleanest illustration that *ablation dominance is not baseline
dominance* — a model can be internally coherent and still worthless.

### 5. Protocol inflation (~1.5 p) — **core contribution**

Table 1 from `PROTOCOL_INFLATION_RESULTS.md`. Each row isolates one convention.

Discussion points, in order of strength:

1. **The cost row carries its own control.** The convention worth +0.232 Sharpe
   to a 56%-per-month-turnover model is worth +0.010 to equal weight. Reporting
   gross returns is not neutral optimism; it is optimism aimed at whichever
   method trades most.
2. **Seed reporting is panel-dependent.** +0.005 on one panel, +0.502 on
   another, same model and allocator. The number of seeds needed to notice is a
   property of the panel, not a constant.
3. **Multiplicity dominates inference.** p 0.359 → 0.680 once the eight other
   searched cells are admitted.
4. **Development-as-held-out is the mildest shortcut here** (+0.090), which is
   worth saying plainly because it is the one the literature discusses most.

Figure 2: inflation per convention, with the equal-weight control bar adjacent
to the dynamic-model bar in the cost panel.

### 6. Pre-registered confirmation (~1 p)

The amendment, the sealed panel, the prediction of H0, and the outcome.
Stress the ordering: prediction committed and hashed *before* the first metric
was read; zero ticker overlap; single-use panel with a stop rule.

Report the full ladder, the paired bootstraps against equal weight and against
the strongest baseline, and the family-wise Reality Check.

**Secondary finding to foreground:** the baseline ranking *inverts* between
panels — equal weight is strongest on the development panel and weakest on the
confirmation panel, while the graph/CVaR controls do the reverse. A study that
fixes one favourite baseline can therefore manufacture either a win or a loss
through that choice alone. This is an argument for reporting a baseline
*ladder*, not a baseline.

### 7. Limitations (~0.5 p)

Fixed-instrument ETF universes, not point-in-time index constituents. Same
provider and overlapping calendar across both panels, so the confirmation is an
independent *instrument* set, not an independent data source. Magnitudes are
specific to these panels and strategies and are not a correction factor to
subtract from published results elsewhere. India NSE identity and
corporate-action work remains incomplete and is excluded from all claims.

### 8. Reproducibility (~0.5 p)

Anonymised artefact: contracts, configs, ledgers, amendments, run manifests,
and the exact commands. Note the independent re-execution datum — a regenerated
run reproduced its documented metrics exactly at a later code revision — and
that the regenerated control was bit-identical to the original.

## Figures and tables budget

| Item | Content | Status |
| --- | --- | --- |
| Figure 1 | Causal decision timeline and leakage controls | to draw |
| Table 1 | Protocol inflation per convention | **locked** |
| Table 2 | Results ladder with development-gate outcomes | assembled from existing docs |
| Table 3 | Track-B confirmation ladder + paired inference | pending seeds |
| Figure 2 | Inflation bar chart with equal-weight control | to draw |
| Table 4 | Release-gate evidence flags | trivial |

## Writing rules for this paper

- No table may imply a strategy works. The honest column of the best inflation
  row is still below equal weight over the same window.
- Every number cites a run id, and every run id resolves to a manifest hash.
- Negative results are stated in the active voice, not buried in hedges.
- No result from `results/legacy`, the survivor-bias Kaggle panel, or the
  community Nifty panel enters a performance table; they appear only as
  documented illustrations of the shortcuts being measured.
