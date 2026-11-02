"""Reprice a canonical target stream under a different one-way cost assumption.

Target weights are copied from an immutable canonical source run; this command
does not refit a model, rebuild a graph, or create new signals.  It is
therefore appropriate for transaction-cost sensitivity only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.backtest import BacktestConfig, run_backtest
from src.research.data import validate_price_panel
from src.research.evaluation import performance_metrics
from src.research.reproducibility import (
    input_paths_with_existing_source_sidecars,
    sha256_file,
    write_run_manifest,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", required=True, help="Canonical run directory whose targets are repriced")
    parser.add_argument("--prices", required=True, help="Validated close/OHLCV panel used by the source run")
    parser.add_argument("--config", required=True, help="Original strategy config that fixes execution and cap rules")
    parser.add_argument("--transaction-cost-bps", required=True, type=float)
    parser.add_argument(
        "--min-rebalance-turnover",
        type=float,
        default=0.0,
        help="Skip a scheduled trade when realised one-way turnover is below this band",
    )
    parser.add_argument(
        "--first-signal-date",
        help="Optional inclusive start date for a development, validation, or test target slice",
    )
    parser.add_argument(
        "--last-signal-date",
        help="Optional inclusive end date for a development, validation, or test target slice",
    )
    args = parser.parse_args()
    if args.transaction_cost_bps < 0:
        raise ValueError("transaction-cost-bps cannot be negative")

    source_directory = _project_path(args.source_run)
    source_manifest_path = source_directory / "run_manifest.json"
    source_targets_path = source_directory / "targets.csv"
    if not source_manifest_path.exists() or not source_targets_path.exists():
        raise FileNotFoundError("source-run must contain run_manifest.json and targets.csv")
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("kind") != "canonical-run-manifest":
        raise ValueError("source-run must be a canonical manifest-backed run")
    config = _read_json(_project_path(args.config))
    prices_path = _project_path(args.prices)
    prices, price_report = validate_price_panel(pd.read_csv(prices_path), require_ohlcv=False)
    targets = pd.read_csv(source_targets_path, parse_dates=["signal_date"])
    if args.first_signal_date is not None:
        targets = targets.loc[targets["signal_date"] >= pd.Timestamp(args.first_signal_date)].copy()
    if args.last_signal_date is not None:
        targets = targets.loc[targets["signal_date"] <= pd.Timestamp(args.last_signal_date)].copy()
    if targets.empty:
        raise ValueError("No source targets remain after applying signal-date bounds")
    backtest = BacktestConfig(
        execution_lag_sessions=int(config["execution_lag_sessions"]),
        transaction_cost_bps=args.transaction_cost_bps,
        max_weight=float(config["max_weight"]),
        min_rebalance_turnover=args.min_rebalance_turnover,
    )
    evaluation_prices = prices
    if args.last_signal_date is not None:
        evaluation_prices = prices.loc[
            prices["date"] <= pd.Timestamp(args.last_signal_date)
        ].copy()
    result = run_backtest(evaluation_prices, targets, config=backtest)
    metrics = performance_metrics(
        result.daily["net_return"], annualization_factor=int(config.get("annualization_factor", 252))
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars(
            [prices_path, source_manifest_path, source_targets_path]
        ),
        extra={
            "analysis_kind": "fixed_target_transaction_cost_sensitivity",
            "source_run_id": source_manifest.get("run_id"),
            "source_git_revision": source_manifest.get("git_revision"),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "transaction_cost_bps": args.transaction_cost_bps,
            "min_rebalance_turnover": backtest.min_rebalance_turnover,
            "first_signal_date": args.first_signal_date,
            "last_signal_date": args.last_signal_date,
            "evaluation_end_date": args.last_signal_date,
            "execution_lag_sessions": backtest.execution_lag_sessions,
            "max_weight": backtest.max_weight,
            "price_panel": {
                "n_rows": price_report.n_rows,
                "n_assets": price_report.n_assets,
                "start": price_report.start.isoformat(),
                "end": price_report.end.isoformat(),
            },
        },
    )
    output_directory = manifest_path.parent
    targets.to_csv(output_directory / "targets.csv", index=False)
    result.daily.to_csv(output_directory / "daily_nav.csv")
    result.holdings.to_csv(output_directory / "holdings.csv", index=False)
    result.executions.to_csv(output_directory / "executions.csv", index=False)
    (output_directory / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved fixed-target cost sensitivity run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
