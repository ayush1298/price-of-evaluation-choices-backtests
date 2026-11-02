"""Run a causal HMM, soft-regime-mixture portfolio experiment.

This is the canonical replacement for the legacy hard K-Means regime switch.
It requires point-in-time prices and point-in-time membership history, fits a
Gaussian HMM only on earlier monthly observations, then blends state-specific
graph/allocation targets by the current HMM posterior.
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

from src.research.backtest import (
    BacktestConfig,
    WalkForwardConfig,
    rebalance_signal_dates,
    run_walk_forward,
)
from src.research.data import validate_membership_history, validate_price_panel
from src.research.evaluation import performance_metrics
from src.research.experiments import (
    BaselineSignalBuilder,
    BaselineStrategyConfig,
    ProbabilisticRegimeRouter,
)
from src.research.regimes import ExpandingGaussianHMMRegime, build_market_regime_features
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def _read_json(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        return json.load(stream)


def _strategy_from_config(
    state_config: dict[str, object],
    canonical: dict[str, object],
) -> BaselineStrategyConfig:
    minimum_assets = int(state_config.get("minimum_assets", canonical["minimum_eligible_assets"]))
    return BaselineStrategyConfig(
        graph_method=str(state_config.get("graph_method", "pcc")),  # type: ignore[arg-type]
        selector=str(state_config.get("selector", "all")),  # type: ignore[arg-type]
        allocator=str(state_config.get("allocator", "equal_weight")),  # type: ignore[arg-type]
        lookback_sessions=int(state_config.get("lookback_sessions", canonical["minimum_history_sessions"])),
        minimum_assets=minimum_assets,
        portfolio_size=(
            int(state_config["portfolio_size"])
            if "portfolio_size" in state_config
            else minimum_assets
        ),
        edge_fraction=float(state_config.get("edge_fraction", 0.002)),
        max_weight=float(state_config.get("max_weight", canonical["max_weight"])),
        cvar_alpha=float(state_config.get("cvar_alpha", 0.05)),
        cvar_aversion=float(state_config.get("cvar_aversion", 1.0)),
        turnover_penalty=float(state_config.get("turnover_penalty", 0.0)),
    )


def _serializable_price_report(report: object) -> dict[str, object]:
    data = asdict(report)  # type: ignore[arg-type]
    data["start"] = report.start.isoformat()  # type: ignore[attr-defined]
    data["end"] = report.end.isoformat()  # type: ignore[attr-defined]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time close-price CSV")
    parser.add_argument("--universe", required=True, help="Point-in-time membership CSV with date,symbol,eligible")
    parser.add_argument("--config", default="configs/canonical_us_regime_hmm.json")
    parser.add_argument("--market-symbol", help="Optional benchmark symbol present in the prices input")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transaction-cost-bps", type=float, help="Optional one-way cost override")
    parser.add_argument(
        "--first-signal-date",
        help="Optional inclusive outer-evaluation boundary; earlier observations remain available to the expanding HMM",
    )
    parser.add_argument(
        "--last-signal-date",
        help="Optional inclusive final signal date for a development or validation window",
    )
    args = parser.parse_args()

    canonical = _read_json(args.config)
    regime = canonical.get("regime")
    if not isinstance(regime, dict) or regime.get("model") != "expanding_gaussian_hmm":
        raise ValueError("Config must contain a regime.model of 'expanding_gaussian_hmm'")
    features = regime.get("features")
    state_configs = regime.get("state_strategies")
    if not isinstance(features, list) or not all(isinstance(column, str) for column in features):
        raise ValueError("regime.features must be a non-empty list of column names")
    if not isinstance(state_configs, dict):
        raise ValueError("regime.state_strategies must map HMM state IDs to strategy configurations")

    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    universe = validate_membership_history(pd.read_csv(args.universe))
    market_features = build_market_regime_features(prices, market_symbol=args.market_symbol)
    signal_dates = rebalance_signal_dates(prices, str(canonical["rebalance_frequency"]))
    monthly_features = market_features.loc[market_features["date"].isin(signal_dates)].copy()
    hmm = ExpandingGaussianHMMRegime(
        n_components=int(regime["n_components"]),
        min_train_size=int(regime["minimum_training_observations"]),
        random_state=args.seed,
    )
    probabilities = hmm.predict_expanding(
        monthly_features,
        features,
        ordering_feature=str(regime["ordering_feature"]),
    )
    if probabilities.empty:
        raise ValueError("No HMM state probabilities were produced; provide a longer pre-test history")
    builders: dict[int, BaselineSignalBuilder] = {}
    for state in range(hmm.n_components):
        raw_config = state_configs.get(str(state))
        if not isinstance(raw_config, dict):
            raise ValueError(f"Missing dictionary strategy configuration for HMM state {state}")
        builders[state] = BaselineSignalBuilder(_strategy_from_config(raw_config, canonical), universe)
    router = ProbabilisticRegimeRouter(probabilities, builders, max_weight=float(canonical["max_weight"]))
    first_regime_date = probabilities["date"].min()
    sessions_to_first_regime = int((prices["date"].drop_duplicates() <= first_regime_date).sum())
    walk_forward = WalkForwardConfig(
        minimum_history_sessions=max(int(canonical["minimum_history_sessions"]), sessions_to_first_regime),
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
        max_weight=float(canonical["max_weight"]),
    )
    result = run_walk_forward(
        prices,
        router,
        walk_forward_config=walk_forward,
        backtest_config=backtest,
    )
    metrics = performance_metrics(
        result.backtest.daily["net_return"], annualization_factor=int(canonical["annualization_factor"])
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars([args.prices, args.universe]),
        extra={
            "price_panel": _serializable_price_report(price_report),
            "hmm": {
                "n_components": hmm.n_components,
                "min_train_size": hmm.min_train_size,
                "random_state": hmm.random_state,
                "feature_columns": features,
                "market_symbol": args.market_symbol,
            },
            "walk_forward": asdict(walk_forward),
            "backtest": asdict(backtest),
            "state_strategies": {state: asdict(builder.config) for state, builder in builders.items()},
        },
    )
    output_directory = manifest_path.parent
    result.targets.to_csv(output_directory / "targets.csv", index=False)
    result.backtest.daily.to_csv(output_directory / "daily_nav.csv")
    result.backtest.holdings.to_csv(output_directory / "holdings.csv", index=False)
    result.backtest.executions.to_csv(output_directory / "executions.csv", index=False)
    market_features.to_csv(output_directory / "market_regime_features_daily.csv", index=False)
    probabilities.to_csv(output_directory / "regime_probabilities.csv", index=False)
    router.diagnostics_frame().to_csv(output_directory / "regime_router_diagnostics.csv", index=False)
    state_diagnostics = pd.concat(
        [builder.diagnostics_frame().assign(regime_state=state) for state, builder in builders.items()],
        ignore_index=True,
    )
    state_diagnostics.to_csv(output_directory / "state_graph_diagnostics.csv", index=False)
    (output_directory / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical HMM regime-mixture run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
