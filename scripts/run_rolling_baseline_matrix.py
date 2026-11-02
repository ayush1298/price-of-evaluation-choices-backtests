"""Evaluate frozen conventional baselines over multiple bounded time windows.

The command does not tune a strategy per fold.  A single JSON file fixes the
base execution policy, each allocation/graph choice, and every inclusive
signal boundary.  Inputs remain available as pre-decision history, while the
backtest is truncated at each window's final signal date.  This provides a
reproducible stability check complementary to one chronological outer split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.backtest import BacktestConfig, WalkForwardConfig, run_walk_forward
from src.research.data import validate_membership_history, validate_price_panel
from src.research.evaluation import performance_metrics
from src.research.experiments import BaselineSignalBuilder, BaselineStrategyConfig
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def _read_json(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Rolling-matrix configuration must be a JSON object")
    return payload


def _path_from_config(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _window_items(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("windows must be a non-empty list")
    windows: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for value in payload:
        if not isinstance(value, dict):
            raise ValueError("Every window must be an object")
        identifier = value.get("id")
        first_signal_date = value.get("first_signal_date")
        last_signal_date = value.get("last_signal_date")
        if not all(isinstance(item, str) and item for item in (identifier, first_signal_date, last_signal_date)):
            raise ValueError("Every window requires non-empty id, first_signal_date, and last_signal_date")
        if identifier in identifiers:
            raise ValueError(f"Duplicate window id: {identifier}")
        if pd.Timestamp(first_signal_date) > pd.Timestamp(last_signal_date):
            raise ValueError(f"Window {identifier} ends before it begins")
        identifiers.add(identifier)
        windows.append(
            {
                "id": identifier,
                "first_signal_date": first_signal_date,
                "last_signal_date": last_signal_date,
            }
        )
    return windows


def _strategy_items(payload: object, base_config: dict[str, object]) -> list[tuple[str, BaselineStrategyConfig]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("strategies must be a non-empty list")
    strategies: list[tuple[str, BaselineStrategyConfig]] = []
    identifiers: set[str] = set()
    defaults = {
        "lookback_sessions": int(base_config["minimum_history_sessions"]),
        "minimum_assets": int(base_config["minimum_eligible_assets"]),
        "portfolio_size": int(base_config["minimum_eligible_assets"]),
        "edge_fraction": float(base_config.get("edge_fraction", 0.002)),
        "max_weight": float(base_config["max_weight"]),
        "cvar_alpha": 0.05,
        "cvar_aversion": 1.0,
        "turnover_penalty": 0.0,
    }
    for value in payload:
        if not isinstance(value, dict):
            raise ValueError("Every strategy must be an object")
        identifier = value.get("id")
        settings = value.get("strategy")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every strategy requires a non-empty id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate strategy id: {identifier}")
        if not isinstance(settings, dict):
            raise ValueError(f"Strategy {identifier} requires a strategy object")
        allowed = {"graph_method", "selector", "allocator", *defaults}
        unexpected = sorted(set(settings).difference(allowed))
        if unexpected:
            raise ValueError(f"Strategy {identifier} has unsupported fields: {unexpected}")
        merged = {**defaults, **settings}
        for required in ("graph_method", "selector", "allocator"):
            if required not in merged:
                raise ValueError(f"Strategy {identifier} is missing {required}")
        strategies.append((identifier, BaselineStrategyConfig(**merged)))
        identifiers.add(identifier)
    return strategies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time close-price CSV")
    parser.add_argument("--universe", required=True, help="Point-in-time membership CSV")
    parser.add_argument("--config", required=True, help="Frozen rolling-matrix JSON")
    args = parser.parse_args()

    matrix_config = _read_json(args.config)
    base_config_path = matrix_config.get("base_config")
    if not isinstance(base_config_path, str) or not base_config_path:
        raise ValueError("Rolling-matrix configuration requires base_config")
    base_config = _read_json(_path_from_config(base_config_path))
    windows = _window_items(matrix_config.get("windows"))
    strategies = _strategy_items(matrix_config.get("strategies"), base_config)
    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    universe = validate_membership_history(pd.read_csv(args.universe))
    backtest = BacktestConfig(
        execution_lag_sessions=int(base_config["execution_lag_sessions"]),
        transaction_cost_bps=float(base_config["transaction_cost_bps"]),
        max_weight=float(base_config["max_weight"]),
    )

    metric_rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for window in windows:
        for strategy_id, strategy in strategies:
            builder = BaselineSignalBuilder(strategy, universe)
            walk_forward = WalkForwardConfig(
                minimum_history_sessions=max(int(base_config["minimum_history_sessions"]), strategy.lookback_sessions + 1),
                rebalance_frequency=str(base_config["rebalance_frequency"]),
                first_signal_date=window["first_signal_date"],
                last_signal_date=window["last_signal_date"],
            )
            result = run_walk_forward(
                prices,
                builder,
                walk_forward_config=walk_forward,
                backtest_config=backtest,
            )
            metrics = performance_metrics(
                result.backtest.daily["net_return"], annualization_factor=int(base_config["annualization_factor"])
            )
            metric_rows.append(
                {
                    "window_id": window["id"],
                    "strategy_id": strategy_id,
                    "first_signal_date": window["first_signal_date"],
                    "last_signal_date": window["last_signal_date"],
                    **metrics,
                }
            )
            daily = result.backtest.daily.reset_index().loc[:, ["date", "net_return"]]
            daily.insert(0, "strategy_id", strategy_id)
            daily.insert(0, "window_id", window["id"])
            daily_rows.append(daily)
            diagnostics = builder.diagnostics_frame()
            diagnostics.insert(0, "strategy_id", strategy_id)
            diagnostics.insert(0, "window_id", window["id"])
            diagnostic_rows.append(diagnostics)

    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars([args.prices, args.universe, base_config_path]),
        extra={
            "base_config": base_config_path,
            "price_panel": {
                **asdict(price_report),
                "start": price_report.start.isoformat(),
                "end": price_report.end.isoformat(),
            },
            "backtest": asdict(backtest),
            "windows": windows,
            "strategies": [{"id": identifier, **asdict(strategy)} for identifier, strategy in strategies],
        },
    )
    output_directory = manifest_path.parent
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_directory / "metrics_by_window.csv", index=False)
    metrics_frame.groupby("strategy_id", sort=True).agg(
        mean_sharpe=("sharpe", "mean"),
        standard_deviation_sharpe=("sharpe", "std"),
        mean_annualized_return=("annualized_return", "mean"),
        mean_max_drawdown=("max_drawdown", "mean"),
        windows=("window_id", "count"),
    ).reset_index().to_csv(output_directory / "strategy_summary.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(output_directory / "daily_returns_by_window.csv", index=False)
    pd.concat(diagnostic_rows, ignore_index=True).to_csv(output_directory / "graph_diagnostics_by_window.csv", index=False)
    print(f"Saved rolling baseline matrix to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
