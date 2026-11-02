"""Emit the manuscript's LaTeX tables directly from generated result artefacts.

A paper about transcription-sensitive reporting should not retype its own
numbers.  Every table below is rendered from the JSON the pipeline produced, so
a changed run propagates to the manuscript on the next build instead of leaving
a stale figure in a table nobody re-checked.

Outputs go to paper/tables/ as standalone .tex fragments that main.tex inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def _metrics(run_id: str) -> dict | None:
    path = PROJECT_ROOT / "results" / "canonical" / run_id / "performance_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mean_turnover(run_id: str) -> float | None:
    """Mean realised one-way turnover per executed rebalance."""
    import csv

    path = PROJECT_ROOT / "results" / "canonical" / run_id / "daily_nav.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        values = [
            float(row["turnover"])
            for row in csv.DictReader(stream)
            if row.get("rebalanced", "").strip().lower() in {"true", "1"}
        ]
    return sum(values) / len(values) if values else None


def _escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def table_protocol_inflation() -> str:
    payload = _read("results/protocol_inflation_table_v1.json")

    # Group the cells by the choice they isolate. Scattered rows make the claim
    # "six choices across fourteen cells" impossible to verify by eye, and hide
    # the within-choice spread that is the paper's point about non-uniformity.
    groups = [
        ("Development boundary", ["development-window-as-held-out",
                                  "development-window-as-held-out-20-epochs"]),
        ("Seed selection", ["best-seed-us30", "best-seed-us50",
                            "best-seed-trackb-2020"]),
        ("Trading costs", ["costless-execution-dynamic",
                           "costless-execution-equal-weight-control",
                           "costless-execution-dynamic-trackb",
                           "costless-execution-equal-weight-trackb"]),
        ("Benchmark choice", ["benchmark-choice-trackb"]),
        ("Date alignment", ["period-mismatch-quarterly-trackb"]),
        ("Search multiplicity", ["search-multiplicity", "search-multiplicity-trackb",
                                 "search-multiplicity-trackb-2020"]),
        ("Combined choices", ["compound-three-shortcuts"]),
    ]
    # What the two columns actually contrast, per row.
    READING = {
        "development-window-as-held-out": "development vs held out",
        "development-window-as-held-out-20-epochs": "development vs held out",
        "best-seed-us30": "best seed vs median",
        "best-seed-us50": "best seed vs median",
        "best-seed-trackb-2020": "best seed vs median",
        "costless-execution-dynamic": "gross vs net of 10\\,bps",
        "costless-execution-equal-weight-control": "gross vs net of 10\\,bps",
        "costless-execution-dynamic-trackb": "gross vs net of 10\\,bps",
        "costless-execution-equal-weight-trackb": "gross vs net of 10\\,bps",
        "compound-three-shortcuts": "three choices at once vs adjusted",
    }
    by_id = {}
    for key in ("rows", "benchmark_choice_rows", "period_mismatch_rows", "inference_rows"):
        for row in payload.get(key, []):
            by_id[row["row_id"]] = (key, row)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Measured inflation from each evaluation choice, grouped by the choice",
        r"isolated. Each row reads the same runs twice, holding the panel, strategy, allocator,",
        r"cost model and execution rule fixed. Rows marked \emph{sealed} were computed on the",
        r"confirmation panel, which had never been examined when the earlier rows were produced.",
        r"Inflation is the reported value minus the adjusted one, computed from unrounded values;",
        r"columns are rounded to three decimals, so a difference taken from them can disagree in the",
        r"last digit. The search-multiplicity group reports $p$-values rather than Sharpe",
        r"ratios, so its inflation is a difference in $p$. Sharpe cells aggregate seeds by the",
        r"estimator the frozen protocol declared, mean at five epochs and median in the",
        r"declared-budget experiment.}",
        r"\label{tab:inflation}",
        r"\small",
        r"\begin{tabular}{p{2.6cm}p{5.0cm}p{4.2cm}rrr}",
        r"\toprule",
        r"Choice & Panel / strategy & Reading & Reported & Adjusted & Inflation \\",
        r"\midrule",
    ]
    for index, (label, ids) in enumerate(groups):
        present = [i for i in ids if i in by_id]
        if not present:
            continue
        if index:
            lines.append(r"\addlinespace[2pt]")
        for position, row_id in enumerate(present):
            kind, row = by_id[row_id]
            first = _escape(label) if position == 0 else ""
            # Display names must match the other tables; the JSON keeps the
            # longer provenance string.
            DISPLAY = {"search-multiplicity":
                       "HMM cond.\\ risk (2, 252), 9-cell grid vs matched HRP-756 control"}
            panel = f"{_escape(row['panel'])}: {DISPLAY.get(row_id) or _escape(row['strategy'])}"
            if kind == "rows":
                reading = READING.get(row_id, "reported vs adjusted")
                a = f"${row['reported']['value']:.3f}$"
                b = f"${row['honest']['value']:.3f}$"
                inf = f"$\\mathbf{{{row['inflation']:+.3f}}}$"
            elif kind == "inference_rows":
                reading = "best cell vs corrected"
                a = f"${row['reported']['p_value']:.3f}$"
                b = f"${row['honest']['p_value']:.3f}$"
                inf = f"$\\mathbf{{{row['p_value_understatement']:+.3f}}}$"
            else:
                reading = ("one baseline vs ladder" if kind == "benchmark_choice_rows"
                           else "own dates vs matched")
                a = f"${row['reported_edge']:+.3f}$"
                b = f"${row['honest_edge']:+.3f}$"
                flag = r"\,$\dagger$" if row.get("conclusion_sign_flips") else ""
                inf = f"$\\mathbf{{{row['inflation']:+.3f}}}${flag}"
            lines.append(f"{first} & {panel} & {reading} & {a} & {b} & {inf} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]",
        r"\footnotesize $\dagger$ the sign of the conclusion changes, not merely its magnitude.",
        r"\end{table*}",
    ]
    return "\n".join(lines)


CANDIDATE_LADDER = [
    # (method, comparator, statistic, outcome) -- each figure is the one recorded
    # in the corresponding results document; see the docs/ references there.
    ("Soft-regime cond.\\ covariance", "matched uncond.\\ risk", "$0.909$"),
    ("HMM cond.\\ risk (2, 252)", "matched HRP-756", "$0.359$"),
    ("\\quad same, corrected for the 9-cell search", "matched HRP-756", "$0.680$"),
    ("Group-aware cond.\\ covariance", "matched global-HMM risk", "$0.732$"),
    ("Edge-masked graph attention", "similarity passing", "$0.474$"),
    ("IC training objective (US)", "likelihood-only", "$0.298$"),
    ("IC training objective (India)", "likelihood-only", "$0.998$"),
    ("Probability-gated regime experts", "regime-free model", "$0.719$"),
    ("Direct probability concat.", "regime-free model", "$0.519$"),
    ("VIX-conditioned regime model", "regime-free model", "$0.717$"),
    ("Dynamic graph-temporal, full", "own graph-only ablation", "$0.007$"),
    ("\\quad same", "own temporal-only abl.", "$0.023$"),
    ("\\quad same", "equal weight", "$0.705$"),
]


def table_candidate_ladder() -> str:
    """Every proposed candidate and the paired test it failed."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Every proposed candidate, its matched comparator, and the one-sided",
        r"paired moving-block $p$-value for a positive advantage. The final three rows",
        r"are the same model, which beats both of its own ablations and still loses",
        r"to equal weight. Thirteen tests cover nine candidates. The",
        r"information-coefficient objective appears on two panels, the HMM entry appears with",
        r"and without its search correction, and the dynamic model is tested against two",
        r"ablations and one baseline. All rows resolve to manifest-backed runs (Section 3).}",
        r"\label{tab:ladder}",
        r"\footnotesize",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Candidate & Compared against & $p$ \\",
        r"\midrule",
    ]
    for method, comparator, stat in CANDIDATE_LADDER:
        lines.append(f"{method} & {comparator} & {stat} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


SPECIFICATION = [
    ("Soft-regime cond.\\ covariance",
     "expanding soft $k$-means, 3 states, on four market features; state-mixed shrinkage covariance, 756-session lookback",
     "matched unconditional risk: the same HRP allocation on an unconditional covariance",
     "isolates the state mixture", "1"),
    ("HMM cond.\\ risk (2, 252)",
     "expanding Gaussian HMM, 2 states, 252-session covariance lookback, entropy fallback to unconditional risk",
     "matched HRP-756, the stronger unconditional control; the lookback-matched HRP-252 gives $p{=}0.209$",
     "isolates state conditioning", "1"),
    ("9-cell HMM grid",
     "the same model over states $\\in\\{2,3,4\\}$ and lookback $\\in\\{63,126,252\\}$ sessions",
     "matched HRP-756",
     "prices the search over those two axes", "1"),
    ("Group-aware cond.\\ covariance",
     "as the HMM candidate, but state conditioning retained only within five ex-ante asset groups",
     "matched global-HMM risk",
     "isolates cross-group conditioning", "1"),
    ("Edge-masked graph attention",
     "attention over the sparsified graph in place of similarity message passing",
     "similarity passing: the same encoder with fixed weights",
     "isolates learned edge weighting", "5"),
    ("IC training objective",
     "ranking-aligned loss on the cross-sectional information coefficient",
     "likelihood-only: the same model under a Gaussian likelihood",
     "isolates the training objective", "5"),
    ("Regime-conditioned variants",
     "regime probabilities entering the model by direct concatenation, by gated expert heads, or through a VIX covariate",
     "regime-free model: the same architecture without state input",
     "isolates regime conditioning", "5"),
    ("Dynamic graph-temporal",
     "partial-correlation graph, 252-session lookback, 5\\% edges; 2 graph layers into a 2-layer, 4-head Transformer, $d{=}48$, sequence length 12",
     "its graph-only and temporal-only ablations, and equal weight",
     "isolates each stage, then the whole model",
     "5"),
]


def table_specification() -> str:
    """One row per candidate family: what it is, and what its comparison isolates."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Candidate specifications: mechanism, matched comparator, and what each",
        r"comparison isolates. Every candidate sends its output through the shared allocator of",
        r"Section~\ref{sec:apparatus}, so a comparison differs only in the component named.",
        r"Seeds are the prescribed set $\{7, 17, 29, 43, 59\}$ where a method is stochastic.}",
        r"\label{tab:spec}",
        r"\footnotesize",
        r"\begin{tabular}{p{2.7cm}p{5.1cm}p{4.0cm}p{2.9cm}r}",
        r"\toprule",
        r"Family & Mechanism & Matched comparator & Isolates & Seeds \\",
        r"\midrule",
    ]
    for name, mech, comp, iso, seeds in SPECIFICATION:
        lines.append(f"{name} & {mech} & {comp} & {iso} & {seeds} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_confirmation_ladder() -> str:
    rows = [
        ("alt-etf-confirm-partial-hub-avoid-cvar", "Partial-corr.\\ hub-avoid, CVaR", False),
        ("alt-etf-confirm-pcc-hub-avoid-cvar", "PCC hub-avoid, CVaR", False),
        ("alt-etf-confirm-inverse-vol", "Inverse volatility", False),
        ("alt-etf-confirm-hrp-252", "HRP (252)", False),
        ("alt-etf-confirm-hmm-risk-h2-l252-b0", "HMM cond.\\ risk (2, 252)", True),
        ("alt-etf-confirm-hrp-756", "HRP (756)", False),
        ("alt-etf-confirm-soft-regime-conditional-hrp", "Soft-regime cond.\\ covariance", True),
        ("alt-etf-confirm-equal-weight", "Equal weight", False),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Confirmation-panel ladder. Net of 10\,bps one-way costs, 10\% cap,",
        r"one-session execution lag, monthly rebalance, first signal 2014-01-31.",
        r"Turnover is mean realised one-way turnover per execution. Italicised rows",
        r"are the pre-registered candidates. No candidate leads, nor does any buy rank",
        r"by taking more drawdown, tail, or trading risk.}",
        r"\label{tab:confirmation}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Strategy & Sharpe & Max DD & CVaR$_{95}$ & Turn. \\",
        r"\midrule",
    ]
    for run_id, label, is_candidate in rows:
        m = _metrics(run_id)
        if m is None:
            continue
        name = rf"\emph{{{label}}}" if is_candidate else label
        sharpe = f"{m['sharpe']:.3f}"
        if run_id.endswith("partial-hub-avoid-cvar"):
            sharpe = rf"\textbf{{{sharpe}}}"
        turnover = _mean_turnover(run_id)
        turnover_cell = "---" if turnover is None else f"{turnover*100:.1f}\\%"
        lines.append(
            f"{name} & {sharpe} & ${m['max_drawdown']*100:.1f}$\\% & "
            f"${m['cvar_95_daily']*100:.2f}$\\% & {turnover_cell} \\\\"
        )
    seeds = [_metrics(f"alt-etf-confirm-dynamic-seed{s}") for s in (7, 17, 29, 43, 59)]
    values = sorted(m["sharpe"] for m in seeds if m)
    lines += [
        r"\midrule",
        rf"\emph{{Dynamic, 5-seed median}} & {values[len(values)//2]:.3f} & --- & --- & 48.9\% \\",
        rf"\quad seed range & {values[0]:.3f}--{values[-1]:.3f} & --- & --- & --- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_robustness() -> str:
    rows = [
        ("partial-hub-avoid-cvar", "Partial-corr., CVaR", False),
        ("pcc-hub-avoid-cvar", "PCC hub-avoid, CVaR", False),
        ("inverse-vol", "Inverse volatility", False),
        ("hrp-252", "HRP (252)", False),
        ("hrp-756", "HRP (756)", False),
        ("hmm-risk-h2-l252-b0", "HMM cond.\\ risk", True),
        ("soft-regime-conditional-hrp", "Soft-regime cond.\\ covariance", True),
        ("equal-weight", "Equal weight", False),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cap and rebalance-frequency robustness (net Sharpe). A 2\% cap is",
        r"arithmetically impossible on a 30-asset universe and is not attempted.",
        r"Both regime candidates fail outright under quarterly rebalancing.}",
        r"\label{tab:robustness}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Strategy & 10\% mon. & 5\% mon. & 10\% qtr. \\",
        r"\midrule",
    ]
    for suffix, label, is_candidate in rows:
        base, cap5, quarterly = (
            _metrics(f"alt-etf-confirm-{suffix}"),
            _metrics(f"alt-etf-cap5-{suffix}"),
            _metrics(f"alt-etf-quarterly-{suffix}"),
        )
        name = rf"\emph{{{label}}}" if is_candidate else label
        fmt = lambda m: f"{m['sharpe']:.3f}" if m else r"\textbf{fails}"
        lines.append(f"{name} & {fmt(base)} & {fmt(cap5)} & {fmt(quarterly)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_period_mismatch() -> str:
    matched = [
        ("alt-etf-quarterly-dynamic-seed7", "Dynamic, seed 7", True),
        ("alt-etf-quarterly2020-hrp-252", "HRP (252)", False),
        ("alt-etf-quarterly2020-inverse-vol", "Inverse volatility", False),
        ("alt-etf-quarterly2020-equal-weight", "Equal weight", False),
        ("alt-etf-quarterly2020-pcc-hub-avoid-cvar", "PCC hub-avoid, CVaR", False),
        ("alt-etf-quarterly2020-partial-hub-avoid-cvar", "Partial-corr., CVaR", False),
        ("alt-etf-quarterly2020-hrp-756", "HRP (756)", False),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The quarterly cell, read two ways. The model's 24-quarter warm-up starts it",
        r"in 2020 while the baselines start in 2014; every run is individually correct.",
        r"Re-running the baselines over the model's own window collapses a $+0.54$ edge to",
        r"$+0.045$ against equal weight and $+0.017$ against the strongest matched baseline.",
        r"The model realises 17 fewer sessions than the baselines because its first execution",
        r"follows its first signal by one session and its warm-up completes intra-quarter.}",
        r"\label{tab:period}",
        r"\footnotesize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Strategy (quarterly) & Sharpe & Obs. \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\emph{As printed: benchmarks from 2014, model from 2020}} \\",
    ]
    for run_id, label in (
        ("alt-etf-quarterly-dynamic-seed7", "Dynamic, seed 7"),
        ("alt-etf-quarterly-equal-weight", "Equal weight"),
    ):
        m = _metrics(run_id)
        lines.append(f"\\quad {label} & {m['sharpe']:.3f} & {int(m['n_observations'])} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\emph{Date-matched: all runs from 2020-03-31}} \\")
    for run_id, label, bold in matched:
        m = _metrics(run_id)
        if m is None:
            continue
        value = rf"\textbf{{{m['sharpe']:.3f}}}" if bold else f"{m['sharpe']:.3f}"
        lines.append(f"\\quad {label} & {value} & {int(m['n_observations'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/tables")
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "inflation.tex": table_protocol_inflation,
        "specification.tex": table_specification,
        "ladder.tex": table_candidate_ladder,
        "confirmation.tex": table_confirmation_ladder,
        "robustness.tex": table_robustness,
        "period_mismatch.tex": table_period_mismatch,
    }
    for name, builder in builders.items():
        (output_dir / name).write_text(builder() + "\n", encoding="utf-8")
        print(f"wrote {args.output_dir}/{name}")


if __name__ == "__main__":
    main()
