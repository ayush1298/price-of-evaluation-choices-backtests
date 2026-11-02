"""Run the dynamic graph-transformer candidate in a strict walk-forward loop.

Each decision trains a fresh model only on graph windows whose *forward label*
ended before that decision date, forecasts the current graph window, passes
those scores to the constrained mean-CVaR allocator, and backtests the targets
with the canonical daily cost-aware engine.
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

from src.research.backtest import BacktestConfig, rebalance_signal_dates
from src.research.data import (
    resolve_contract_provenance,
    validate_membership_history,
    validate_price_panel,
)
from src.research.evaluation import performance_metrics
from src.research.experiments import DynamicGraphExperimentConfig, run_dynamic_graph_walk_forward
from src.research.graphs import GraphSnapshotConfig
from src.research.models import DynamicGraphTrainingConfig, NodeFeatureConfig
from src.research.regimes import ExpandingSoftKMeansRegime, build_market_regime_features
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def _read_json(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        return json.load(stream)


def _input_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _price_report_metadata(report: object) -> dict[str, object]:
    data = asdict(report)  # type: ignore[arg-type]
    data["start"] = report.start.isoformat()  # type: ignore[attr-defined]
    data["end"] = report.end.isoformat()  # type: ignore[attr-defined]
    return data


def _experiment_config(
    canonical: dict[str, object],
    *,
    seed: int,
    device: str,
    epochs: int | None,
    first_signal_date: str | None,
    last_signal_date: str | None,
    transaction_cost_bps: float | None,
) -> DynamicGraphExperimentConfig:
    graph = canonical["graph"]
    node_features = canonical["node_features"]
    model = canonical["model"]
    allocation = canonical["allocation"]
    if not all(isinstance(item, dict) for item in (graph, node_features, model, allocation)):
        raise ValueError("graph, node_features, model, and allocation config sections must be dictionaries")
    regime_expert_heads = int(model.get("regime_expert_heads", 0))
    if regime_expert_heads:
        regime = canonical.get("regime_conditioning")
        if not isinstance(regime, dict):
            raise ValueError("regime_expert_heads requires a regime_conditioning configuration")
        if regime_expert_heads != int(regime.get("n_components", 0)):
            raise ValueError("regime_expert_heads must equal regime_conditioning.n_components")
    graph_config = GraphSnapshotConfig(
        method=str(graph["method"]),  # type: ignore[arg-type]
        lookback_sessions=int(graph["lookback_sessions"]),
        edge_fraction=float(graph["edge_fraction"]),
        minimum_fit_assets=int(canonical["minimum_eligible_assets"]),
    )
    training_config = DynamicGraphTrainingConfig(
        model_dimension=int(model["model_dimension"]),
        attention_heads=int(model["attention_heads"]),
        temporal_layers=int(model["temporal_layers"]),
        graph_layers=int(model["graph_layers"]),
        dropout=float(model["dropout"]),
        epochs=epochs if epochs is not None else int(model["epochs"]),
        batch_size=int(model["batch_size"]),
        learning_rate=float(model["learning_rate"]),
        weight_decay=float(model["weight_decay"]),
        gradient_clip_norm=float(model["gradient_clip_norm"]),
        regime_expert_heads=regime_expert_heads,
        graph_encoder=str(model.get("graph_encoder", "similarity")),
        information_coefficient_loss_weight=float(model.get("information_coefficient_loss_weight", 0.0)),
        random_seed=seed,
        device=device,
    )
    return DynamicGraphExperimentConfig(
        rebalance_frequency=str(canonical["rebalance_frequency"]),
        annualization_factor=int(canonical["annualization_factor"]),
        minimum_eligible_assets=int(canonical["minimum_eligible_assets"]),
        max_weight=float(canonical["max_weight"]),
        graph=graph_config,
        node_features=NodeFeatureConfig(
            lookback_sessions=int(node_features["lookback_sessions"]),
            annualization_factor=int(canonical["annualization_factor"]),
        ),
        sequence_length=int(model["sequence_length"]),
        minimum_completed_training_windows=int(model["minimum_completed_training_windows"]),
        training=training_config,
        allocation_lookback_sessions=int(allocation["lookback_sessions"]),
        cvar_alpha=float(allocation["cvar_alpha"]),
        cvar_aversion=float(allocation["cvar_aversion"]),
        turnover_penalty=float(allocation["turnover_penalty"]),
        backtest=BacktestConfig(
            execution_lag_sessions=int(canonical["execution_lag_sessions"]),
            transaction_cost_bps=(
                transaction_cost_bps
                if transaction_cost_bps is not None
                else float(canonical["transaction_cost_bps"])
            ),
            max_weight=float(canonical["max_weight"]),
        ),
        first_signal_date=first_signal_date,
        last_signal_date=last_signal_date,
    )


def _regime_probabilities_from_config(
    canonical: dict[str, object],
    prices: pd.DataFrame,
    *,
    last_signal_date: str | None,
    external_covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    raw = canonical.get("regime_conditioning")
    if raw is None:
        return None, None
    if not isinstance(raw, dict) or raw.get("model") != "expanding_soft_kmeans":
        raise ValueError("regime_conditioning must specify model='expanding_soft_kmeans'")
    features = raw.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(column, str) for column in features):
        raise ValueError("regime_conditioning.features must be a non-empty string list")
    ordering_feature = raw.get("ordering_feature")
    if not isinstance(ordering_feature, str):
        raise ValueError("regime_conditioning.ordering_feature must be a string")
    market_symbol = raw.get("market_symbol")
    if market_symbol is not None and not isinstance(market_symbol, str):
        raise ValueError("regime_conditioning.market_symbol must be a string when supplied")
    market_features = build_market_regime_features(
        prices, market_symbol=market_symbol, external_covariates=external_covariates
    )
    missing_features = set(features).difference(market_features.columns)
    if missing_features:
        raise ValueError(f"regime_conditioning.features are unavailable: {sorted(missing_features)}")
    signal_dates = rebalance_signal_dates(prices, str(canonical["rebalance_frequency"]))
    monthly_features = market_features.loc[market_features["date"].isin(signal_dates)].copy()
    if last_signal_date is not None:
        monthly_features = monthly_features.loc[
            monthly_features["date"] <= pd.Timestamp(last_signal_date)
        ].copy()
    model = ExpandingSoftKMeansRegime(
        n_components=int(raw.get("n_components", 3)),
        min_train_size=int(raw.get("minimum_training_observations", 36)),
        random_state=int(raw.get("random_state", 7)),
        temperature=float(raw.get("temperature", 1.0)),
    )
    probabilities = model.predict_expanding(
        monthly_features,
        features,
        ordering_feature=ordering_feature,
    )
    if probabilities.empty:
        raise ValueError("No causal soft-regime probabilities were produced; extend market history")
    metadata = {
        "model": "expanding_soft_kmeans",
        "n_components": model.n_components,
        "min_train_size": model.min_train_size,
        "random_state": model.random_state,
        "temperature": model.temperature,
        "features": features,
        "ordering_feature": ordering_feature,
        "market_symbol": market_symbol,
        "external_covariate_columns": (
            [column for column in external_covariates.columns if column != "date"]
            if external_covariates is not None
            else []
        ),
    }
    return probabilities, metadata


def _project_path(value: str | Path) -> Path:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time close-price CSV")
    parser.add_argument("--universe", required=True, help="Point-in-time membership CSV with date,symbol,eligible")
    parser.add_argument("--config", default="configs/canonical_us_dynamic_graph_transformer.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda")
    parser.add_argument("--epochs", type=int, help="Optional explicit epoch override recorded in the run manifest")
    parser.add_argument("--transaction-cost-bps", type=float, help="Optional one-way cost override")
    parser.add_argument(
        "--regime-covariates",
        help="Optional daily CSV with date plus finite external market-state covariates; exact dates only",
    )
    parser.add_argument(
        "--first-signal-date",
        help="Optional inclusive outer-evaluation boundary; earlier labels remain eligible for each expanding training fit",
    )
    parser.add_argument(
        "--last-signal-date",
        help="Optional inclusive final signal date for a development or validation window",
    )
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


    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    membership = validate_membership_history(pd.read_csv(args.universe))
    covariate_path = _input_path(args.regime_covariates) if args.regime_covariates else None
    external_covariates = pd.read_csv(covariate_path) if covariate_path is not None else None
    experiment_config = _experiment_config(
        canonical,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        first_signal_date=args.first_signal_date,
        last_signal_date=args.last_signal_date,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    regime_probabilities, regime_metadata = _regime_probabilities_from_config(
        canonical,
        prices,
        last_signal_date=args.last_signal_date,
        external_covariates=external_covariates,
    )
    result = run_dynamic_graph_walk_forward(
        prices,
        membership,
        config=experiment_config,
        regime_probabilities=regime_probabilities,
    )
    metrics = performance_metrics(
        result.backtest.daily["net_return"], annualization_factor=experiment_config.annualization_factor
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars(
            [
                args.prices,
                args.universe,
                *([covariate_path] if covariate_path is not None else []),
                *contract_input_paths,
            ]
        ),
        extra={
            "price_panel": _price_report_metadata(price_report),
            **({} if contract_metadata is None else {"data_contract": contract_metadata}),
            # Recorded in the same shape as every other canonical runner so this
            # model is comparable under multiplicity-aware inference.
            "walk_forward": {
                "rebalance_frequency": experiment_config.rebalance_frequency,
                "first_signal_date": experiment_config.first_signal_date,
                "last_signal_date": experiment_config.last_signal_date,
            },
            "backtest": asdict(experiment_config.backtest),
            "master_universe_date": result.master_universe_date.isoformat(),
            "master_universe_size": len(result.master_universe),
            "training": asdict(experiment_config.training),
            "first_signal_date": experiment_config.first_signal_date,
            "last_signal_date": experiment_config.last_signal_date,
            "regime_conditioning": regime_metadata,
            "regime_covariates_path": str(covariate_path.relative_to(PROJECT_ROOT)) if covariate_path else None,
            "dataset": {
                "n_samples": len(result.dataset),
                "sequence_length": result.dataset.features.shape[1],
                "feature_columns": list(result.dataset.feature_columns),
                "minimum_completed_training_windows": experiment_config.minimum_completed_training_windows,
            },
            "graph_date_coverage": {
                "candidate_dates": int(len(result.graph_date_coverage)),
                "buildable_dates": int(result.graph_date_coverage["eligible_for_graph"].sum()),
                "retained_for_model_dates": int(result.graph_date_coverage["retained_for_model"].sum()),
            },
        },
    )
    output_directory = manifest_path.parent
    result.targets.to_csv(output_directory / "targets.csv", index=False)
    result.backtest.daily.to_csv(output_directory / "daily_nav.csv")
    result.backtest.holdings.to_csv(output_directory / "holdings.csv", index=False)
    result.backtest.executions.to_csv(output_directory / "executions.csv", index=False)
    result.graph_snapshots.diagnostics.to_csv(output_directory / "graph_diagnostics.csv", index=False)
    result.graph_date_coverage.to_csv(output_directory / "graph_date_coverage.csv", index=False)
    result.node_features.to_csv(output_directory / "node_features.csv", index=False)
    if result.regime_probabilities is not None:
        result.regime_probabilities.to_csv(output_directory / "regime_probabilities.csv", index=False)
    result.decision_diagnostics.to_csv(output_directory / "model_training_diagnostics.csv", index=False)
    (output_directory / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical dynamic-graph run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
