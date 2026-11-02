"""Run the causal soft-regime conditional-HRP portfolio candidate.

At each monthly close an expanding soft K-Means model infers the observed
market state.  Historical daily covariance estimates are probability-weighted
by only the state known at each earlier close, mixed using the current state's
probabilities, and passed to the shared capped HRP allocator.  No return
forecast, future regime state, or future membership is used.
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

from src.research.backtest import BacktestConfig, WalkForwardConfig, rebalance_signal_dates, run_walk_forward
from src.research.data import (
    resolve_contract_provenance,
    validate_membership_history,
    validate_price_panel,
)
from src.research.evaluation import performance_metrics
from src.research.experiments import RegimeConditionalRiskConfig, SoftRegimeConditionalHRPBuilder
from src.research.regimes import ExpandingSoftKMeansRegime, build_market_regime_features
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def _read_json(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    return payload


def _project_path(value: str | Path) -> Path:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time OHLCV/close-price CSV")
    parser.add_argument("--universe", required=True, help="Point-in-time membership CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transaction-cost-bps", type=float, help="Optional one-way cost override")
    parser.add_argument("--first-signal-date", help="Inclusive evaluation-start signal boundary")
    parser.add_argument("--last-signal-date", help="Inclusive final development/evaluation signal date")
    parser.add_argument(
        "--data-contract",
        help=(
            "Optional validated versioned data contract JSON. Required when this run must be "
            "comparable with contract-aware candidates under multiplicity-aware inference."
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

    regime = canonical.get("regime")
    risk = canonical.get("conditional_risk")
    if not isinstance(regime, dict) or regime.get("model") != "expanding_soft_kmeans":
        raise ValueError("Config must contain regime.model = 'expanding_soft_kmeans'")
    if not isinstance(risk, dict):
        raise ValueError("Config must contain a conditional_risk object")
    features = regime.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(column, str) for column in features):
        raise ValueError("regime.features must be a non-empty string list")

    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    universe = validate_membership_history(pd.read_csv(args.universe))
    market_features = build_market_regime_features(prices)
    signal_dates = rebalance_signal_dates(prices, str(canonical["rebalance_frequency"]))
    monthly_features = market_features.loc[market_features["date"].isin(signal_dates)].copy()
    model = ExpandingSoftKMeansRegime(
        n_components=int(regime["n_components"]),
        min_train_size=int(regime["minimum_training_observations"]),
        random_state=args.seed,
        temperature=float(regime.get("temperature", 1.0)),
    )
    probabilities = model.predict_expanding(
        monthly_features,
        features,
        ordering_feature=str(regime["ordering_feature"]),
    )
    if probabilities.empty:
        raise ValueError("No soft-regime probabilities were produced; provide a longer history")
    risk_config = RegimeConditionalRiskConfig(
        lookback_sessions=int(risk["lookback_sessions"]),
        minimum_assets=int(canonical["minimum_eligible_assets"]),
        minimum_effective_observations=float(risk["minimum_effective_observations"]),
        covariance_shrinkage=float(risk["covariance_shrinkage"]),
        max_weight=float(canonical["max_weight"]),
    )
    builder = SoftRegimeConditionalHRPBuilder(risk_config, universe, probabilities)
    first_probability_date = pd.Timestamp(probabilities["date"].min())
    sessions_to_first_probability = int((prices["date"].drop_duplicates() <= first_probability_date).sum())
    walk_forward = WalkForwardConfig(
        minimum_history_sessions=max(
            int(canonical["minimum_history_sessions"]),
            risk_config.lookback_sessions + 1,
            sessions_to_first_probability,
        ),
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
        max_weight=risk_config.max_weight,
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
            "price_panel": {**asdict(price_report), "start": price_report.start.isoformat(), "end": price_report.end.isoformat()},
            **({} if contract_metadata is None else {"data_contract": contract_metadata}),
            "soft_kmeans": {
                "n_components": model.n_components,
                "min_train_size": model.min_train_size,
                "temperature": model.temperature,
                "random_state": model.random_state,
                "feature_columns": features,
            },
            "conditional_risk": asdict(risk_config),
            "walk_forward": asdict(walk_forward),
            "backtest": asdict(backtest),
        },
    )
    output_directory = manifest_path.parent
    result.targets.to_csv(output_directory / "targets.csv", index=False)
    result.backtest.daily.to_csv(output_directory / "daily_nav.csv")
    result.backtest.holdings.to_csv(output_directory / "holdings.csv", index=False)
    result.backtest.executions.to_csv(output_directory / "executions.csv", index=False)
    market_features.to_csv(output_directory / "market_regime_features_daily.csv", index=False)
    probabilities.to_csv(output_directory / "regime_probabilities.csv", index=False)
    builder.diagnostics_frame().to_csv(output_directory / "conditional_risk_diagnostics.csv", index=False)
    (output_directory / "performance_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved canonical soft-regime conditional-HRP run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
