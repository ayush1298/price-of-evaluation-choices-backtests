"""Run one point-in-time, walk-forward graph/allocation baseline.

Unlike the legacy scripts, this command requires an explicit historical
membership file and emits daily NAV, holdings, turnover, costs, and per-signal
graph diagnostics into a fresh canonical run directory.
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
from src.research.data import (
    resolve_contract_provenance,
    validate_membership_history,
    validate_price_panel,
)
from src.research.evaluation import performance_metrics
from src.research.experiments import BaselineSignalBuilder, BaselineStrategyConfig
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def _read_json(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        return json.load(stream)


def _project_path(value: str | Path) -> Path:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time close-price CSV")
    parser.add_argument(
        "--universe",
        required=True,
        help="Point-in-time membership CSV with date,symbol,eligible",
    )
    parser.add_argument("--config", default="configs/canonical_us_monthly.json")
    parser.add_argument("--graph", choices=["pcc", "partial_correlation"], default="pcc")
    parser.add_argument("--selector", choices=["all", "hub_avoid", "hub_follow"], default="all")
    parser.add_argument(
        "--allocator",
        choices=["equal_weight", "inverse_volatility", "hierarchical_risk_parity", "mean_cvar"],
        default="equal_weight",
    )
    parser.add_argument("--lookback-sessions", type=int)
    parser.add_argument("--minimum-assets", type=int)
    parser.add_argument("--portfolio-size", type=int)
    parser.add_argument("--edge-fraction", type=float, help="Defaults to config edge_fraction or 0.002")
    parser.add_argument("--max-weight", type=float)
    parser.add_argument("--cvar-alpha", type=float, default=0.05)
    parser.add_argument("--cvar-aversion", type=float, default=1.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--transaction-cost-bps", type=float, help="Optional one-way cost override")
    parser.add_argument(
        "--first-signal-date",
        help="Optional inclusive outer-evaluation boundary; earlier observations remain available as training history",
    )
    parser.add_argument(
        "--last-signal-date",
        help="Optional inclusive final signal date for a development or validation window",
    )
    parser.add_argument(
        "--data-contract",
        help=(
            "Optional validated versioned data contract JSON. Required when this run must be "
            "comparable with contract-aware candidates, for example as a Reality Check benchmark."
        ),
    )
    parser.add_argument(
        "--data-contract-report",
        help="Saved contract-validation report JSON; required whenever --data-contract is supplied",
    )
    args = parser.parse_args()

    if bool(args.data_contract) != bool(args.data_contract_report):
        parser.error("--data-contract and --data-contract-report must be supplied together")

    canonical = _read_json(args.config)
    contract_metadata: dict[str, object] | None = None
    contract_input_paths: list[str] = []
    if args.data_contract:
        contract_metadata, contract_input_paths = resolve_contract_provenance(
            _project_path(args.data_contract),
            _project_path(args.data_contract_report),
            config=canonical,
            project_root=PROJECT_ROOT,
        )

    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    universe = validate_membership_history(pd.read_csv(args.universe))
    minimum_assets = args.minimum_assets or int(canonical["minimum_eligible_assets"])
    strategy = BaselineStrategyConfig(
        graph_method=args.graph,
        selector=args.selector,
        allocator=args.allocator,
        lookback_sessions=args.lookback_sessions or int(canonical["minimum_history_sessions"]),
        minimum_assets=minimum_assets,
        portfolio_size=args.portfolio_size if args.portfolio_size is not None else minimum_assets,
        edge_fraction=args.edge_fraction if args.edge_fraction is not None else float(canonical.get("edge_fraction", 0.002)),
        max_weight=args.max_weight if args.max_weight is not None else float(canonical["max_weight"]),
        cvar_alpha=args.cvar_alpha,
        cvar_aversion=args.cvar_aversion,
        turnover_penalty=args.turnover_penalty,
    )
    builder = BaselineSignalBuilder(strategy, universe)
    walk_forward = WalkForwardConfig(
        minimum_history_sessions=max(int(canonical["minimum_history_sessions"]), strategy.lookback_sessions + 1),
        rebalance_frequency=str(canonical["rebalance_frequency"]),
        first_signal_date=args.first_signal_date,
        last_signal_date=args.last_signal_date,
    )
    backtest = BacktestConfig(
        execution_lag_sessions=int(canonical["execution_lag_sessions"]),
        transaction_cost_bps=(
            args.transaction_cost_bps
            if args.transaction_cost_bps is not None
            else float(canonical["transaction_cost_bps"])
        ),
        max_weight=strategy.max_weight,
    )
    result = run_walk_forward(
        prices,
        builder,
        walk_forward_config=walk_forward,
        backtest_config=backtest,
    )
    metrics = performance_metrics(
        result.backtest.daily["net_return"], annualization_factor=int(canonical["annualization_factor"])
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars(
            [args.prices, args.universe, *contract_input_paths]
        ),
        extra={
            "price_panel": {
                **asdict(price_report),
                "start": price_report.start.isoformat(),
                "end": price_report.end.isoformat(),
            },
            **({} if contract_metadata is None else {"data_contract": contract_metadata}),
            "strategy": asdict(strategy),
            "walk_forward": asdict(walk_forward),
            "backtest": asdict(backtest),
        },
    )
    output_directory = manifest_path.parent
    result.targets.to_csv(output_directory / "targets.csv", index=False)
    result.backtest.daily.to_csv(output_directory / "daily_nav.csv")
    result.backtest.holdings.to_csv(output_directory / "holdings.csv", index=False)
    result.backtest.executions.to_csv(output_directory / "executions.csv", index=False)
    builder.diagnostics_frame().to_csv(output_directory / "graph_diagnostics.csv", index=False)
    (output_directory / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical baseline run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
