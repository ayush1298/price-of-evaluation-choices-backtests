"""Report deflated Sharpe for a candidate family against its own search size.

The deflated Sharpe asks a different question from the Reality Check: not
"could the best candidate's *edge over a benchmark* be search noise", but
"could the best candidate's *Sharpe itself* be search noise, given how many
configurations were tried".  Both are needed, because a family can contain a
candidate with a respectable standalone Sharpe that still has no edge, and a
candidate with an edge whose Sharpe is unremarkable.

The trial set must be the complete family.  Passing only the survivors would
understate the search and inflate the diagnostic, which is the exact failure
this script exists to expose, so it refuses an incomplete ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evaluation import audit_validation_candidate_ledger
from src.research.evaluation.candidate_ledger import load_candidate_ledger
from src.research.evaluation.inference import deflated_sharpe_ratio
from src.research.reproducibility import sha256_file


def _daily_net_returns(run_directory: Path) -> pd.Series:
    daily = pd.read_csv(run_directory / "daily_nav.csv", parse_dates=["date"]).set_index("date")
    return pd.to_numeric(daily["net_return"], errors="raise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger", required=True)
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument(
        "--extra-trial-run",
        action="append",
        default=[],
        help="Additional run id belonging to the same search family, for example a baseline ladder member",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ledger_path = PROJECT_ROOT / args.candidate_ledger
    results_root = PROJECT_ROOT / args.results_root
    output = PROJECT_ROOT / args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")

    ledger = load_candidate_ledger(ledger_path)
    audit = audit_validation_candidate_ledger(ledger, results_root=results_root)
    if audit.incomplete_candidate_count:
        raise ValueError(
            "Deflated Sharpe requires every candidate in the ledger to have matching artifacts; "
            "a reduced trial set would understate the search"
        )

    run_ids = [str(c["run_id"]) for c in ledger["candidates"]] + list(args.extra_trial_run)
    returns = {run_id: _daily_net_returns(results_root / run_id) for run_id in run_ids}

    # Trial Sharpes must be on one periodicity; use daily, matching the series
    # the diagnostic is applied to.
    trials = {
        run_id: float(series.mean() / series.std(ddof=1))
        for run_id, series in returns.items()
    }
    best = max(trials, key=trials.get)
    result = deflated_sharpe_ratio(returns[best], list(trials.values()))

    payload = {
        "kind": "deflated-sharpe-diagnostic",
        "candidate_ledger": {"path": str(ledger_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(ledger_path)},
        "n_trials": len(trials),
        "trial_daily_sharpes": trials,
        "best_run_id": best,
        "result": result.as_dict() if hasattr(result, "as_dict") else vars(result),
        "warning": (
            "The trial set must be the complete search family. This diagnostic addresses "
            "configuration mining only; it is not an untouched outer test."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Trials: {len(trials)}  best: {best}")
    print(json.dumps(payload["result"], indent=2))
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
