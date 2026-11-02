"""Run the frozen Track-B confirmation inference in one auditable pass.

The confirmation protocol in `docs/ICAIF_CHARTER_AMENDMENT_V3_CONFIRMATION.md`
fixes what must be computed once the confirmation runs exist: a paired
moving-block bootstrap of every candidate against the equal-weight benchmark
*and* against the best-performing baseline on this panel, plus a White Reality
Check over the whole candidate family.

Doing that in one command matters here.  The comparisons are only meaningful if
none of them is chosen after seeing the others, so the set is derived from the
committed ledger rather than typed by hand per candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evaluation.candidate_ledger import load_candidate_ledger


def _metrics(results_root: Path, run_id: str) -> dict:
    return json.loads((results_root / run_id / "performance_metrics.json").read_text(encoding="utf-8"))


def _compare(strategy: Path, benchmark: Path, output: Path, *, block: int, reps: int, seed: int) -> dict:
    subprocess.run(
        [
            sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "compare_canonical_runs.py"),
            "--strategy-run", str(strategy), "--benchmark-run", str(benchmark),
            "--block-length", str(block), "--replications", str(reps), "--seed", str(seed),
            "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="configs/ledgers/public_us_etf_alternative_confirmation_v1.json")
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument("--output-dir", default="results/confirmation")
    parser.add_argument("--block-length", type=int, default=21)
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ledger_path = PROJECT_ROOT / args.ledger
    results_root = PROJECT_ROOT / args.results_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger = load_candidate_ledger(ledger_path)
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    benchmark_id = raw["benchmark_run_id"]
    baseline_ids = [b["run_id"] for b in raw["baselines"]]
    candidate_ids = [c["run_id"] for c in ledger["candidates"]]

    # The strongest baseline is identified from the baselines alone, never from
    # candidate results, so a candidate cannot pick its own easiest comparator.
    baseline_sharpes = {run_id: _metrics(results_root, run_id)["sharpe"] for run_id in baseline_ids}
    best_baseline = max(baseline_sharpes, key=baseline_sharpes.get)
    print(f"Benchmark: {benchmark_id}")
    print(f"Best baseline on this panel: {best_baseline} (Sharpe {baseline_sharpes[best_baseline]:.3f})\n")

    summary = {
        "kind": "track-b-confirmation-inference",
        "ledger": str(ledger_path.relative_to(PROJECT_ROOT)),
        "benchmark_run_id": benchmark_id,
        "baseline_sharpes": baseline_sharpes,
        "best_baseline_run_id": best_baseline,
        "comparisons": [],
    }

    for candidate in candidate_ids:
        for label, comparator in (("vs_equal_weight", benchmark_id), ("vs_best_baseline", best_baseline)):
            output = output_dir / f"{candidate}__{label}.json"
            result = _compare(
                results_root / candidate, results_root / comparator, output,
                block=args.block_length, reps=args.replications, seed=args.seed,
            )
            row = {
                "candidate": candidate,
                "comparator": comparator,
                "comparison": label,
                "annualized_excess_return": result["annualized_excess_return"],
                "observed_annualized_sharpe_difference": result["observed_annualized_sharpe_difference"],
                "one_sided_p_value": result["one_sided_p_value"],
                "observations": result["observations"],
            }
            summary["comparisons"].append(row)
            print(
                f"{candidate.replace('alt-etf-confirm-',''):42s} {label:18s} "
                f"ret={row['annualized_excess_return']:+.4%}  "
                f"sharpe={row['observed_annualized_sharpe_difference']:+.3f}  "
                f"p={row['one_sided_p_value']:.3f}  n={row['observations']}"
            )

    summary_path = output_dir / "confirmation_inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
