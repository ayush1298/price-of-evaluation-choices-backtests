"""Leakage-safe walk-forward evaluation for the dynamic graph model candidate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.allocation import mean_cvar_weights
from src.research.backtest import BacktestConfig, BacktestResult, rebalance_signal_dates, run_backtest
from src.research.data import eligible_universe, validate_membership_history, validate_price_panel
from src.research.graphs import GraphSnapshotConfig, GraphSnapshotResult, build_graph_snapshots
from src.research.models import (
    DEFAULT_NODE_FEATURES,
    DynamicGraphTrainingConfig,
    DynamicGraphWindowDataset,
    NodeFeatureConfig,
    build_dynamic_graph_window_dataset,
    build_node_feature_snapshots,
    build_rebalance_forward_returns,
    completed_training_mask,
    predict_dynamic_graph_forecast,
    train_expanding_dynamic_graph_transformer,
)
from src.research.regimes import broadcast_regime_probabilities_to_nodes


@dataclass(frozen=True)
class DynamicGraphExperimentConfig:
    """Explicit choices for one dynamic-graph model candidate experiment."""

    rebalance_frequency: str
    annualization_factor: int
    minimum_eligible_assets: int
    max_weight: float
    graph: GraphSnapshotConfig
    node_features: NodeFeatureConfig
    sequence_length: int
    minimum_completed_training_windows: int
    training: DynamicGraphTrainingConfig
    allocation_lookback_sessions: int
    cvar_alpha: float
    cvar_aversion: float
    turnover_penalty: float
    backtest: BacktestConfig
    first_signal_date: str | None = None
    last_signal_date: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_eligible_assets < 2:
            raise ValueError("minimum_eligible_assets must be at least two")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.minimum_eligible_assets * self.max_weight < 1 - 1e-12:
            raise ValueError("minimum_eligible_assets and max_weight cannot form a fully invested portfolio")
        if self.sequence_length < 1 or self.minimum_completed_training_windows < 1:
            raise ValueError("sequence_length and minimum_completed_training_windows must be positive")
        if self.allocation_lookback_sessions < 2:
            raise ValueError("allocation_lookback_sessions must be at least two")
        if not 0 < self.cvar_alpha < 1 or self.cvar_aversion < 0 or self.turnover_penalty < 0:
            raise ValueError("Invalid mean-CVaR allocation settings")
        for field_name, value in (
            ("first_signal_date", self.first_signal_date),
            ("last_signal_date", self.last_signal_date),
        ):
            if value is None:
                continue
            try:
                pd.Timestamp(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field_name} must be a parseable date") from error
        if self.first_signal_date is not None and self.last_signal_date is not None:
            if pd.Timestamp(self.first_signal_date) > pd.Timestamp(self.last_signal_date):
                raise ValueError("first_signal_date cannot be after last_signal_date")


@dataclass(frozen=True)
class DynamicGraphExperimentResult:
    """All outputs needed for a reproducible proposed-model run."""

    targets: pd.DataFrame
    backtest: BacktestResult
    graph_snapshots: GraphSnapshotResult
    graph_date_coverage: pd.DataFrame
    node_features: pd.DataFrame
    dataset: DynamicGraphWindowDataset
    decision_diagnostics: pd.DataFrame
    master_universe: tuple[str, ...]
    master_universe_date: pd.Timestamp
    regime_probabilities: pd.DataFrame | None


def run_dynamic_graph_walk_forward(
    prices: pd.DataFrame,
    universe_history: pd.DataFrame,
    *,
    config: DynamicGraphExperimentConfig,
    regime_probabilities: pd.DataFrame | None = None,
) -> DynamicGraphExperimentResult:
    """Run strict expanding training, scoring, allocation, and daily backtest.

    The function precomputes causal feature/graph snapshots for convenience,
    but each model fit is strictly filtered to labels whose end date predates
    the current decision. It therefore does not train on the held-out outcome
    of the allocation it is about to submit.
    """
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    membership = validate_membership_history(universe_history)
    trading_dates = pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values()
    all_signal_dates = rebalance_signal_dates(panel, config.rebalance_frequency)
    if config.last_signal_date is not None:
        all_signal_dates = all_signal_dates[all_signal_dates <= pd.Timestamp(config.last_signal_date)]
    graph_signal_dates = pd.DatetimeIndex(
        [date for date in all_signal_dates if (trading_dates <= date).sum() >= config.graph.lookback_sessions + 1]
    )
    graph_date_coverage = _graph_signal_coverage(panel, membership, graph_signal_dates, config.graph)
    graph_signal_dates = pd.DatetimeIndex(
        graph_date_coverage.loc[graph_date_coverage["eligible_for_graph"], "signal_date"].to_numpy()
    )
    if graph_signal_dates.empty:
        raise ValueError(
            "No rebalance date has enough continuously priced eligible assets for the configured graph lookback"
        )
    # The tensor universe is frozen when the model first makes a decision, not
    # at an earlier graph warm-up date.  This avoids excluding assets that are
    # legitimately eligible at the first declared evaluation signal while
    # still preventing later identities from entering the model tensor.
    master_candidates = graph_signal_dates
    if config.first_signal_date is not None:
        master_candidates = master_candidates[master_candidates >= pd.Timestamp(config.first_signal_date)]
    if master_candidates.empty:
        raise ValueError("No buildable graph date exists on or after the first model signal date")
    master_universe_date = pd.Timestamp(master_candidates[0])
    master_universe = tuple(eligible_universe(membership, master_universe_date))
    graph_snapshots = build_graph_snapshots(
        panel,
        membership,
        graph_signal_dates,
        config=config.graph,
    )
    node_features = build_node_feature_snapshots(
        panel,
        membership,
        graph_snapshots.graphs,
        master_universe,
        config=config.node_features,
    )
    feature_columns = DEFAULT_NODE_FEATURES
    if regime_probabilities is not None:
        node_features, regime_feature_columns = broadcast_regime_probabilities_to_nodes(
            node_features, regime_probabilities
        )
        allowed_dates = pd.DatetimeIndex(node_features["date"].unique())
        graph_signal_dates = graph_signal_dates[graph_signal_dates.isin(allowed_dates)]
        if graph_signal_dates.empty:
            raise ValueError("No graph dates remain after causal regime-probability warm-up")
        graph_snapshots = GraphSnapshotResult(
            graphs={date: graph_snapshots.graphs[date] for date in graph_signal_dates},
            diagnostics=graph_snapshots.diagnostics.loc[
                graph_snapshots.diagnostics["signal_date"].isin(graph_signal_dates)
            ].copy(),
        )
        node_features = node_features.loc[node_features["date"].isin(graph_signal_dates)].copy()
        feature_columns = (*DEFAULT_NODE_FEATURES, *regime_feature_columns)
    graph_date_coverage = graph_date_coverage.assign(
        retained_for_model=graph_date_coverage["signal_date"].isin(graph_signal_dates)
    )
    labels = build_rebalance_forward_returns(panel, graph_signal_dates)
    dataset = build_dynamic_graph_window_dataset(
        node_features,
        graph_snapshots.graphs,
        labels,
        master_universe,
        sequence_length=config.sequence_length,
        feature_columns=feature_columns,
    )
    price_matrix = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    previous_weights: pd.Series | None = None
    target_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    first_signal_date = (
        pd.Timestamp(config.first_signal_date) if config.first_signal_date is not None else None
    )
    last_signal_date = (
        pd.Timestamp(config.last_signal_date) if config.last_signal_date is not None else None
    )
    for sample_index, decision_date in enumerate(dataset.input_end_dates):
        if first_signal_date is not None and decision_date < first_signal_date:
            continue
        if last_signal_date is not None and decision_date > last_signal_date:
            continue
        completed = completed_training_mask(dataset, decision_date)
        if int(completed.sum()) < config.minimum_completed_training_windows:
            continue
        training = train_expanding_dynamic_graph_transformer(
            dataset, decision_date, config=config.training
        )
        forecast = predict_dynamic_graph_forecast(training, dataset, decision_date)
        scores = pd.Series(forecast.score.squeeze(0).numpy(), index=master_universe)
        scales = pd.Series(forecast.scale.squeeze(0).numpy(), index=master_universe)
        usable = dataset.target_mask[sample_index].numpy()
        candidates = [symbol for symbol, is_usable in zip(master_universe, usable) if bool(is_usable)]
        returns = _continuous_returns(
            price_matrix, decision_date, candidates, config.allocation_lookback_sessions
        )
        if returns.shape[1] < config.minimum_eligible_assets:
            continue
        weights = mean_cvar_weights(
            returns,
            expected_returns=scores.reindex(returns.columns),
            alpha=config.cvar_alpha,
            cvar_aversion=config.cvar_aversion,
            turnover_penalty=config.turnover_penalty,
            previous_weights=previous_weights,
            max_weight=config.max_weight,
        )
        previous_weights = weights.copy()
        target_rows.extend(
            {"signal_date": decision_date, "symbol": symbol, "weight": float(weight)}
            for symbol, weight in weights.items()
        )
        decision_rows.append(
            {
                "signal_date": decision_date,
                "n_completed_training_windows": int(completed.sum()),
                "latest_training_target_end": max(training.training_target_end_dates),
                "final_training_loss": training.epoch_losses[-1],
                "n_model_usable_assets": len(candidates),
                "n_allocated_assets": len(weights),
                "mean_forecast_score": float(scores.reindex(weights.index).mean()),
                "mean_forecast_scale": float(scales.reindex(weights.index).mean()),
            }
        )
    targets = pd.DataFrame(target_rows, columns=["signal_date", "symbol", "weight"])
    if targets.empty:
        raise ValueError(
            "No model targets were generated. Increase data history or lower graph/model training-window settings."
        )
    # A dynamic universe can legitimately lose the minimum number of usable
    # tensor members after an entry/exit.  Do not let the backtest carry its
    # last model portfolio through later, targetless dates: the realised
    # evaluation must end at the final generated signal (and at any caller's
    # earlier bounded endpoint).
    evaluation_end = pd.Timestamp(targets["signal_date"].max())
    if config.last_signal_date is not None:
        evaluation_end = min(evaluation_end, pd.Timestamp(config.last_signal_date))
    evaluation_panel = panel.loc[panel["date"] <= evaluation_end].copy()
    backtest = run_backtest(evaluation_panel, targets, config=config.backtest)
    return DynamicGraphExperimentResult(
        targets=targets,
        backtest=backtest,
        graph_snapshots=graph_snapshots,
        graph_date_coverage=graph_date_coverage,
        node_features=node_features,
        dataset=dataset,
        decision_diagnostics=pd.DataFrame(decision_rows),
        master_universe=master_universe,
        master_universe_date=master_universe_date,
        regime_probabilities=regime_probabilities,
    )


def _graph_signal_coverage(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    graph_config: GraphSnapshotConfig,
) -> pd.DataFrame:
    """Audit which causal graph dates have enough complete eligible histories.

    Dynamic-universe panels naturally have an early warm-up period and can
    have later listing gaps. Those dates are not repaired or made tradable;
    they are explicitly excluded from graph/model windows before snapshot
    construction. ``build_graph_snapshots`` repeats the same eligibility
    invariant as a defense-in-depth check.
    """
    if signal_dates.empty:
        return pd.DataFrame(
            columns=["signal_date", "n_eligible", "n_with_complete_history", "eligible_for_graph"]
        )
    matrix = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    rows: list[dict[str, object]] = []
    for date in signal_dates:
        eligible = eligible_universe(membership, date)
        window = matrix.loc[matrix.index <= date].reindex(columns=eligible).tail(graph_config.lookback_sessions + 1)
        complete = int(window.notna().all(axis=0).sum())
        rows.append(
            {
                "signal_date": pd.Timestamp(date),
                "n_eligible": len(eligible),
                "n_with_complete_history": complete,
                "eligible_for_graph": complete >= graph_config.minimum_fit_assets,
            }
        )
    return pd.DataFrame(rows)


def _continuous_returns(
    price_matrix: pd.DataFrame,
    date: pd.Timestamp,
    candidates: list[str],
    lookback_sessions: int,
) -> pd.DataFrame:
    window = price_matrix.loc[price_matrix.index <= date].reindex(columns=candidates).tail(lookback_sessions + 1)
    complete = window.columns[window.notna().all()].tolist()
    if not complete:
        return pd.DataFrame()
    returns = window.loc[:, complete].pct_change(fill_method=None).iloc[1:]
    return returns if len(returns) == lookback_sessions else pd.DataFrame()
