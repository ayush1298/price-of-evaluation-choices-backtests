"""Compare two canonical runs with paired block-bootstrap inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.evaluation import paired_moving_block_bootstrap


def _net_returns(run_directory: Path) -> pd.Series:
    manifest_path = run_directory / "run_manifest.json"
    daily_path = run_directory / "daily_nav.csv"
    if not manifest_path.exists() or not daily_path.exists():
        raise FileNotFoundError(f"Canonical run needs run_manifest.json and daily_nav.csv: {run_directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "canonical-run-manifest":
        raise ValueError(f"Not a canonical run directory: {run_directory}")
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date")
    if "net_return" not in daily.columns:
        raise ValueError(f"Daily NAV file has no net_return column: {daily_path}")
    return daily["net_return"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-run", required=True)
    parser.add_argument("--benchmark-run", required=True)
    parser.add_argument("--block-length", type=int, default=21)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    strategy = _net_returns(Path(args.strategy_run))
    benchmark = _net_returns(Path(args.benchmark_run))
    comparison = paired_moving_block_bootstrap(
        strategy,
        benchmark,
        block_length=args.block_length,
        replications=args.replications,
        random_seed=args.seed,
    ).as_dict()
    content = json.dumps(comparison, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
