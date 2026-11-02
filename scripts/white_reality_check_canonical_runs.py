"""Run a White Reality Check over a complete ledger of canonical candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.evaluation import (
    audit_validation_candidate_ledger,
    load_candidate_ledger,
    white_reality_check,
)
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_run(run_directory: Path) -> tuple[str, pd.Series, dict[str, Any]]:
    manifest_path = run_directory / "run_manifest.json"
    daily_path = run_directory / "daily_nav.csv"
    if not manifest_path.exists() or not daily_path.exists():
        raise FileNotFoundError(f"Canonical run needs run_manifest.json and daily_nav.csv: {run_directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid canonical run manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("kind") != "canonical-run-manifest":
        raise ValueError(f"Not a canonical run directory: {run_directory}")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Canonical run has no valid run_id: {run_directory}")
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date")
    if "net_return" not in daily.columns:
        raise ValueError(f"Daily NAV file has no net_return column: {daily_path}")
    returns = pd.to_numeric(daily["net_return"], errors="raise")
    return run_id, returns, manifest


def _comparable_context(manifest: dict[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra")
    if not isinstance(extra, dict):
        raise ValueError(f"Canonical run {manifest.get('run_id')!r} has no extra metadata")
    contract = extra.get("data_contract")
    walk_forward = extra.get("walk_forward")
    backtest = extra.get("backtest")
    if not isinstance(contract, dict) or not isinstance(walk_forward, dict) or not isinstance(backtest, dict):
        raise ValueError(f"Canonical run {manifest.get('run_id')!r} lacks comparable contract/split/backtest metadata")
    return {
        "study_scope": extra.get("study_scope"),
        "data_contract_id": contract.get("contract_id"),
        "first_signal_date": walk_forward.get("first_signal_date"),
        "last_signal_date": walk_forward.get("last_signal_date"),
        "backtest": backtest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger", required=True)
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument("--block-length", type=int, default=21)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ledger_path = _project_path(args.candidate_ledger)
    results_root = _project_path(args.results_root)
    output = _project_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")
    ledger = load_candidate_ledger(ledger_path)
    audit = audit_validation_candidate_ledger(ledger, results_root=results_root)
    if audit.incomplete_candidate_count:
        raise ValueError("White Reality Check requires every candidate in the ledger to have matching artifacts")

    benchmark_id, benchmark_returns, benchmark_manifest = _canonical_run(_project_path(args.benchmark_run))
    benchmark_context = _comparable_context(benchmark_manifest)
    candidate_returns: dict[str, pd.Series] = {}
    candidate_inputs: dict[str, dict[str, str]] = {}
    for candidate in audit.candidates:
        run_id = str(candidate["run_id"])
        candidate_id, returns, manifest = _canonical_run(results_root / run_id)
        if candidate_id != run_id:
            raise ValueError(f"Ledger run ID {run_id!r} does not match manifest run ID {candidate_id!r}")
        if _comparable_context(manifest) != benchmark_context:
            raise ValueError(f"Candidate {run_id!r} does not match benchmark data/split/backtest context")
        candidate_returns[str(candidate["candidate_id"])] = returns
        candidate_inputs[str(candidate["candidate_id"])] = {
            "run_manifest_sha256": sha256_file(results_root / run_id / "run_manifest.json"),
            "daily_nav_sha256": sha256_file(results_root / run_id / "daily_nav.csv"),
        }
    result = white_reality_check(
        pd.DataFrame(candidate_returns),
        benchmark_returns,
        block_length=args.block_length,
        replications=args.replications,
        random_seed=args.seed,
    )
    payload = {
        "kind": "white-reality-check-canonical-runs",
        "candidate_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "candidate_ledger_audit": audit.as_jsonable(),
        "benchmark": {
            "run_id": benchmark_id,
            "run_manifest_sha256": sha256_file(_project_path(args.benchmark_run) / "run_manifest.json"),
            "daily_nav_sha256": sha256_file(_project_path(args.benchmark_run) / "daily_nav.csv"),
        },
        "candidates": candidate_inputs,
        "result": result.as_dict(),
        "warning": (
            "This tests the full ledger family against one matched benchmark. It does not establish data validity, "
            "an untouched outer result, or paper eligibility."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote White Reality Check to {output}")


if __name__ == "__main__":
    main()
