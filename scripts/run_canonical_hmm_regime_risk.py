"""Run the uncertainty-aware HMM conditional-risk allocation candidate.

The runner fits the HMM only on completed earlier monthly observations.  Its
current soft posterior blends state-conditional covariance estimates; diffuse
posteriors continuously fall back toward unconditional risk rather than
causing a hard regime switch.  It is deliberately a risk forecast plus the
shared HRP allocator, not a direct neural portfolio-weight generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.backtest import BacktestConfig, WalkForwardConfig, rebalance_signal_dates, run_walk_forward
from src.research.data import (
    load_data_contract,
    validate_data_contract,
    validate_membership_history,
    validate_price_panel,
    verify_saved_contract_report,
)
from src.research.evaluation import performance_metrics
from src.research.experiments import RegimeConditionalRiskConfig, SoftRegimeConditionalHRPBuilder
from src.research.regimes import ExpandingGaussianHMMRegime, build_market_regime_features
from src.research.reproducibility import (
    input_paths_with_existing_source_sidecars,
    sha256_file,
    write_run_manifest,
)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = _project_path(str(path))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Configuration must be valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    return payload


def _serializable_price_report(report: object) -> dict[str, object]:
    data = asdict(report)  # type: ignore[arg-type]
    data["start"] = report.start.isoformat()  # type: ignore[attr-defined]
    data["end"] = report.end.isoformat()  # type: ignore[attr-defined]
    return data


def _load_asset_groups(canonical: dict[str, Any]) -> tuple[dict[str, str] | None, Path | None, dict[str, object] | None]:
    specification = canonical.get("asset_groups")
    if specification is None:
        return None, None, None
    if not isinstance(specification, dict):
        raise ValueError("asset_groups must be an object when supplied")
    path_value = specification.get("path")
    symbol_column = specification.get("symbol_column", "symbol")
    group_column = specification.get("group_column", "group")
    require_complete = specification.get("require_complete", True)
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("asset_groups.path must be a non-empty string")
    if not isinstance(symbol_column, str) or not isinstance(group_column, str):
        raise ValueError("asset_groups symbol/group columns must be strings")
    if not isinstance(require_complete, bool):
        raise ValueError("asset_groups.require_complete must be a boolean")
    if not require_complete:
        raise ValueError("Group-aware asset-group configurations must require complete symbol coverage")
    path = _project_path(path_value)
    table = pd.read_csv(path)
    if symbol_column not in table.columns or group_column not in table.columns:
        raise ValueError(f"Asset-group file must contain {symbol_column!r} and {group_column!r}")
    selected = table.loc[:, [symbol_column, group_column]].copy()
    selected[symbol_column] = selected[symbol_column].fillna("").astype(str).str.strip().str.upper()
    selected[group_column] = selected[group_column].fillna("").astype(str).str.strip()
    if selected.eq("").any().any() or selected[symbol_column].duplicated().any():
        raise ValueError("Asset-group file must have unique non-empty symbols and groups")
    groups = dict(zip(selected[symbol_column], selected[group_column]))
    metadata = {
        "path": str(path),
        "sha256": sha256_file(path),
        "symbols": int(len(groups)),
        "groups": int(selected[group_column].nunique()),
        "require_complete": require_complete,
    }
    return groups, path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Long-format point-in-time OHLCV/close-price CSV")
    parser.add_argument("--universe", required=True, help="Point-in-time membership CSV")
    parser.add_argument("--data-contract", required=True, help="Validated versioned data contract JSON")
    parser.add_argument("--data-contract-report", required=True, help="Saved contract-validation report JSON")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hmm-components", type=int, help="Validation-grid override recorded in the manifest")
    parser.add_argument("--lookback-sessions", type=int, help="Validation-grid override recorded in the manifest")
    parser.add_argument(
        "--min-rebalance-turnover", type=float, help="Validation-grid override recorded in the manifest"
    )
    parser.add_argument(
        "--minimum-regime-confidence", type=float, help="Validation-grid override recorded in the manifest"
    )
    parser.add_argument("--transaction-cost-bps", type=float, help="Optional one-way cost override")
    parser.add_argument("--first-signal-date", help="Inclusive development or evaluation-start signal boundary")
    parser.add_argument("--last-signal-date", help="Inclusive final development/evaluation signal date")
    args = parser.parse_args()

    canonical = _read_json(args.config)
    regime = canonical.get("regime")
    risk = canonical.get("conditional_risk")
    allocation = canonical.get("allocation", {})
    if not isinstance(regime, dict) or regime.get("model") != "expanding_gaussian_hmm":
        raise ValueError("Config must contain regime.model = 'expanding_gaussian_hmm'")
    if not isinstance(risk, dict):
        raise ValueError("Config must contain a conditional_risk object")
    if not isinstance(allocation, dict):
        raise ValueError("Config allocation must be an object when supplied")
    features = regime.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(column, str) for column in features):
        raise ValueError("regime.features must be a non-empty string list")
    asset_groups, asset_groups_path, asset_groups_metadata = _load_asset_groups(canonical)
    if bool(risk.get("group_aware_within_group_only", False)) and asset_groups is None:
        raise ValueError("Group-aware conditional risk requires an asset_groups configuration")

    contract_path = _project_path(args.data_contract)
    contract = load_data_contract(contract_path)
    contract_report = validate_data_contract(contract, project_root=PROJECT_ROOT)
    verify_saved_contract_report(contract_path, _project_path(args.data_contract_report), contract_report)
    if canonical.get("data_contract_id") != contract_report["contract"]["contract_id"]:
        raise ValueError("Config data_contract_id does not match the supplied data contract")

    prices, price_report = validate_price_panel(pd.read_csv(args.prices), require_ohlcv=False)
    universe = validate_membership_history(pd.read_csv(args.universe))
    market_features = build_market_regime_features(prices)
    signal_dates = rebalance_signal_dates(prices, str(canonical["rebalance_frequency"]))
    monthly_features = market_features.loc[market_features["date"].isin(signal_dates)].copy()
    resolved_components = args.hmm_components if args.hmm_components is not None else int(regime["n_components"])
    resolved_lookback = args.lookback_sessions if args.lookback_sessions is not None else int(risk["lookback_sessions"])
    resolved_turnover_band = (
        args.min_rebalance_turnover
        if args.min_rebalance_turnover is not None
        else float(allocation.get("min_rebalance_turnover", 0.0))
    )
    resolved_confidence = (
        args.minimum_regime_confidence
        if args.minimum_regime_confidence is not None
        else float(risk.get("minimum_regime_confidence", 0.0))
    )
    hmm = ExpandingGaussianHMMRegime(
        n_components=resolved_components,
        min_train_size=int(regime["minimum_training_observations"]),
        random_state=args.seed,
        n_iter=int(regime.get("n_iter", 200)),
        covariance_type=str(regime.get("covariance_type", "diag")),
        startprob_prior=float(regime.get("startprob_prior", 1.1)),
        transmat_prior=float(regime.get("transmat_prior", 1.1)),
    )
    probabilities = hmm.predict_expanding(
        monthly_features,
        features,
        ordering_feature=str(regime["ordering_feature"]),
    )
    if probabilities.empty:
        raise ValueError("No HMM probabilities were produced; provide a longer pre-signal history")
    risk_config = RegimeConditionalRiskConfig(
        lookback_sessions=resolved_lookback,
        minimum_assets=int(canonical["minimum_eligible_assets"]),
        minimum_effective_observations=float(risk["minimum_effective_observations"]),
        covariance_shrinkage=float(risk["covariance_shrinkage"]),
        max_weight=float(canonical["max_weight"]),
        uncertainty_to_unconditional=bool(risk["uncertainty_to_unconditional"]),
        minimum_regime_confidence=resolved_confidence,
        require_causal_training_end=True,
        group_aware_within_group_only=bool(risk.get("group_aware_within_group_only", False)),
    )
    builder = SoftRegimeConditionalHRPBuilder(risk_config, universe, probabilities, asset_groups=asset_groups)
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
        min_rebalance_turnover=resolved_turnover_band,
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
            [
                args.prices,
                args.universe,
                args.data_contract,
                args.data_contract_report,
                *([] if asset_groups_path is None else [asset_groups_path]),
            ]
        ),
        extra={
            "price_panel": _serializable_price_report(price_report),
            "data_contract": {
                "contract_id": contract_report["contract"]["contract_id"],
                "contract_sha256": sha256_file(contract_path),
                "validation_report_sha256": sha256_file(_project_path(args.data_contract_report)),
            },
            "hmm": {
                "n_components": hmm.n_components,
                "min_train_size": hmm.min_train_size,
                "random_state": hmm.random_state,
                "n_iter": hmm.n_iter,
                "covariance_type": hmm.covariance_type,
                "feature_columns": features,
            },
            "conditional_risk": asdict(risk_config),
            "asset_groups": asset_groups_metadata,
            "validation_grid_overrides": {
                "hmm_components": args.hmm_components,
                "lookback_sessions": args.lookback_sessions,
                "min_rebalance_turnover": args.min_rebalance_turnover,
                "minimum_regime_confidence": args.minimum_regime_confidence,
            },
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
    (output_directory / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical HMM conditional-risk run to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
