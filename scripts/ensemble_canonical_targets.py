"""Backtest an ex-ante equal ensemble of aligned canonical target streams."""

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
from src.research.evaluation import equally_average_target_weights, performance_metrics
from src.research.reproducibility import (
    input_paths_with_existing_source_sidecars,
    sha256_file,
    write_run_manifest,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, object]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", action="append", required=True, help="Canonical source run directory; repeat once per member")
    parser.add_argument("--prices", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--last-signal-date",
        help="Optional inclusive evaluation end; required for a bounded development ensemble",
    )
    args = parser.parse_args()
    if len(args.source_run) < 2:
        raise ValueError("At least two --source-run values are required")

    config_path = _project_path(args.config)
    config = _read_json(config_path)
    source_directories = [_project_path(value) for value in args.source_run]
    source_manifests: list[dict[str, object]] = []
    source_manifest_paths: list[Path] = []
    source_target_paths: list[Path] = []
    source_streams: list[pd.DataFrame] = []
    for directory in source_directories:
        manifest_path = directory / "run_manifest.json"
        targets_path = directory / "targets.csv"
        if not manifest_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Canonical source run requires manifest and targets: {directory}")
        manifest = _read_json(manifest_path)
        if manifest.get("kind") != "canonical-run-manifest":
            raise ValueError(f"Not a canonical source run: {directory}")
        source_manifests.append(manifest)
        source_manifest_paths.append(manifest_path)
        source_target_paths.append(targets_path)
        source_streams.append(pd.read_csv(targets_path, parse_dates=["signal_date"]))
    config_hashes = {str(manifest.get("config", {}).get("sha256")) for manifest in source_manifests}
    if len(config_hashes) != 1:
        raise ValueError("Ensemble members must share the same canonical model configuration hash")

    targets = equally_average_target_weights(source_streams)
    prices_path = _project_path(args.prices)
    prices, price_report = validate_price_panel(pd.read_csv(prices_path), require_ohlcv=False)
    evaluation_prices = prices
    if args.last_signal_date is not None:
        evaluation_prices = prices.loc[prices["date"] <= pd.Timestamp(args.last_signal_date)].copy()
    backtest = BacktestConfig(
        execution_lag_sessions=int(config["execution_lag_sessions"]),
        transaction_cost_bps=float(config["transaction_cost_bps"]),
        max_weight=float(config["max_weight"]),
    )
    result = run_backtest(evaluation_prices, targets, config=backtest)
    metrics = performance_metrics(result.daily["net_return"], annualization_factor=int(config["annualization_factor"]))
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=config_path,
        input_paths=input_paths_with_existing_source_sidecars([prices_path, *source_manifest_paths, *source_target_paths]),
        extra={
            "analysis_kind": "equal_target_weight_seed_ensemble",
            "source_run_ids": [manifest.get("run_id") for manifest in source_manifests],
            "source_manifest_sha256": [sha256_file(path) for path in source_manifest_paths],
            "n_members": len(source_streams),
            "first_signal_date": targets["signal_date"].min().isoformat(),
            "last_signal_date": targets["signal_date"].max().isoformat(),
            "evaluation_end_date": args.last_signal_date,
            "execution_lag_sessions": backtest.execution_lag_sessions,
            "transaction_cost_bps": backtest.transaction_cost_bps,
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
    (output_directory / "performance_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved canonical target ensemble to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
