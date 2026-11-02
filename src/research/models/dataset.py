"""Causal data preparation for dynamic graph forecasting experiments.

The utilities here construct model inputs at rebalance dates only.  They do
not choose an investable universe: callers must provide a declared master
universe and a point-in-time membership history.  Ineligible or insufficiently
observed assets are zero-filled and identified by masks rather than silently
dropped, keeping tensor dimensions stable without treating them as tradable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.research.data import eligible_universe, validate_membership_history, validate_price_panel
from src.research.graphs import adjacency_matrix, compute_centrality_scores


DEFAULT_NODE_FEATURES = (
    "one_session_return",
    "trailing_return",
    "realized_volatility",
    "composite_centrality",
    "availability",
)


@dataclass(frozen=True)
class NodeFeatureConfig:
    """Feature windows used at each graph snapshot date."""

    lookback_sessions: int = 21
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        if self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be at least two")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")


@dataclass(frozen=True)
class DynamicGraphWindowDataset(Dataset[dict[str, Tensor]]):
    """Tensor dataset containing causal graph windows and future-return labels."""

    features: Tensor
    adjacency: Tensor
    target_returns: Tensor
    target_mask: Tensor
    input_end_dates: tuple[pd.Timestamp, ...]
    target_end_dates: tuple[pd.Timestamp, ...]
    universe: tuple[str, ...]
    feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 4 or self.adjacency.ndim != 4:
            raise ValueError("features and adjacency must be [samples, time, assets, ...] tensors")
        if self.features.shape[:3] != self.adjacency.shape[:3] or self.adjacency.shape[-1] != self.features.shape[2]:
            raise ValueError("features and adjacency dimensions are incompatible")
        if self.target_returns.shape != self.target_mask.shape:
            raise ValueError("target_returns and target_mask must have the same shape")
        if self.target_returns.shape != (self.features.shape[0], self.features.shape[2]):
            raise ValueError("targets must have [samples, assets] shape")
        if len(self.input_end_dates) != self.features.shape[0]:
            raise ValueError("Every model sample needs one input end date")
        if len(self.target_end_dates) != self.features.shape[0]:
            raise ValueError("Every model sample needs one label-end date")
        if not all(target > input_end for input_end, target in zip(self.input_end_dates, self.target_end_dates)):
            raise ValueError("Every model label must end strictly after its input window")
        if len(self.universe) != self.features.shape[2]:
            raise ValueError("Universe length must match the asset dimension")
        if len(self.feature_columns) != self.features.shape[3]:
            raise ValueError("Feature column count must match the feature dimension")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "adjacency": self.adjacency[index],
            "target_returns": self.target_returns[index],
            "target_mask": self.target_mask[index],
        }


def build_node_feature_snapshots(
    prices: pd.DataFrame,
    universe_history: pd.DataFrame,
    graph_snapshots: Mapping[pd.Timestamp, nx.Graph],
    universe: Sequence[str],
    *,
    config: NodeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Create historical node features and availability flags at graph dates.

    A row is marked ``usable`` only when its asset was point-in-time eligible
    and has a complete feature lookback ending at that signal date. The graph
    centrality itself is still retained for all declared nodes, including
    isolates, because its zero value is informative.
    """
    config = config or NodeFeatureConfig()
    master_universe = _validate_universe(universe)
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    membership = validate_membership_history(universe_history)
    matrix = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    matrix = matrix.reindex(columns=master_universe)
    dates = _snapshot_dates(graph_snapshots, matrix.index)
    rows: list[pd.DataFrame] = []
    for date in dates:
        graph = graph_snapshots[pd.Timestamp(date)]
        eligible = set(eligible_universe(membership, date))
        window = matrix.loc[matrix.index <= date].tail(config.lookback_sessions + 1)
        complete = window.notna().all(axis=0)
        daily_returns = window.pct_change(fill_method=None).iloc[1:]
        one_session = daily_returns.iloc[-1]
        trailing = window.iloc[-1].div(window.iloc[0]).sub(1.0)
        volatility = daily_returns.std(ddof=1).mul(np.sqrt(config.annualization_factor))
        centrality = compute_centrality_scores(graph, universe=master_universe).set_index("ticker")
        usable = complete & pd.Index(master_universe).isin(eligible)
        snapshot = pd.DataFrame(
            {
                "date": pd.Timestamp(date),
                "symbol": master_universe,
                "one_session_return": one_session.reindex(master_universe).to_numpy(),
                "trailing_return": trailing.reindex(master_universe).to_numpy(),
                "realized_volatility": volatility.reindex(master_universe).to_numpy(),
                "composite_centrality": centrality["composite_score"].reindex(master_universe).to_numpy(),
                "availability": usable.astype(float),
                "usable": usable.astype(bool),
            }
        )
        numeric_columns = list(DEFAULT_NODE_FEATURES[:-1])
        snapshot.loc[:, numeric_columns] = snapshot.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
        snapshot.loc[~snapshot["usable"], numeric_columns] = 0.0
        snapshot.loc[:, numeric_columns] = snapshot.loc[:, numeric_columns].fillna(0.0)
        rows.append(snapshot)
    return pd.concat(rows, ignore_index=True)


def build_rebalance_forward_returns(
    prices: pd.DataFrame,
    signal_dates: Sequence[pd.Timestamp],
    *,
    horizon_periods: int = 1,
) -> pd.DataFrame:
    """Build post-signal labels for outer walk-forward model training.

    These labels are intentionally separate from the feature builder. A label
    at ``t`` uses the price at a later rebalance date and therefore must only be
    available to a model-training fold whose evaluation period begins after it.
    """
    if horizon_periods < 1:
        raise ValueError("horizon_periods must be at least one")
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    matrix = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    dates = pd.DatetimeIndex(pd.to_datetime(signal_dates, errors="raise")).sort_values().unique()
    if dates.empty:
        raise ValueError("At least one signal date is required")
    if not dates.isin(matrix.index).all():
        raise ValueError("Every signal date must be an observed price date")
    labels: list[pd.Series] = []
    label_dates: list[pd.Timestamp] = []
    for index in range(len(dates) - horizon_periods):
        current, future = dates[index], dates[index + horizon_periods]
        labels.append(matrix.loc[future].div(matrix.loc[current]).sub(1.0))
        label_dates.append(pd.Timestamp(current))
    output = pd.DataFrame(labels, index=pd.DatetimeIndex(label_dates, name="signal_date"))
    output.attrs["target_end_dates"] = {
        pd.Timestamp(current): pd.Timestamp(future)
        for current, future in zip(label_dates, dates[horizon_periods:])
    }
    return output


def build_dynamic_graph_window_dataset(
    node_features: pd.DataFrame,
    graph_snapshots: Mapping[pd.Timestamp, nx.Graph],
    forward_returns: pd.DataFrame,
    universe: Sequence[str],
    *,
    sequence_length: int = 12,
    feature_columns: Sequence[str] = DEFAULT_NODE_FEATURES,
    dtype: torch.dtype = torch.float32,
) -> DynamicGraphWindowDataset:
    """Pack declared-universe graph snapshots into fixed-shape learning windows."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    master_universe = _validate_universe(universe)
    required = {"date", "symbol", "usable", *feature_columns}
    missing = required.difference(node_features.columns)
    if missing:
        raise ValueError(f"Node features are missing columns: {sorted(missing)}")
    features = node_features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    features["symbol"] = features["symbol"].astype(str)
    if features.duplicated(["date", "symbol"]).any():
        raise ValueError("Node features must contain one row per (date, symbol)")
    dates = _snapshot_dates(graph_snapshots, pd.DatetimeIndex(features["date"].unique()))
    labels = forward_returns.copy()
    labels.index = pd.to_datetime(labels.index, errors="raise")
    labels.columns = labels.columns.astype(str)
    if labels.index.duplicated().any():
        raise ValueError("Forward-return labels must contain one row per signal date")
    target_end_by_signal = labels.attrs.get("target_end_dates")
    if not isinstance(target_end_by_signal, dict):
        raise ValueError(
            "Forward-return labels must come from build_rebalance_forward_returns or provide "
            "a target_end_dates mapping in DataFrame.attrs"
        )

    feature_blocks: list[Tensor] = []
    adjacency_blocks: list[Tensor] = []
    usable_blocks: list[Tensor] = []
    for date in dates:
        snapshot = features.loc[features["date"] == date].set_index("symbol").reindex(master_universe)
        if (
            snapshot.index.isna().any()
            or snapshot.loc[:, feature_columns].isna().any().any()
            or snapshot["usable"].isna().any()
        ):
            raise ValueError(f"Node feature snapshot at {date.date()} does not cover the declared universe")
        values = snapshot.loc[:, feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Node feature snapshot at {date.date()} has non-finite values")
        feature_blocks.append(torch.tensor(values, dtype=dtype))
        adjacency_blocks.append(torch.tensor(adjacency_matrix(graph_snapshots[date], master_universe), dtype=dtype))
        usable_blocks.append(torch.tensor(snapshot["usable"].astype(bool).to_numpy(), dtype=torch.bool))

    samples: list[Tensor] = []
    adjacency_samples: list[Tensor] = []
    target_samples: list[Tensor] = []
    mask_samples: list[Tensor] = []
    end_dates: list[pd.Timestamp] = []
    target_end_dates: list[pd.Timestamp] = []
    for end_index in range(sequence_length - 1, len(dates)):
        date = pd.Timestamp(dates[end_index])
        if date not in labels.index:
            continue
        target_end = target_end_by_signal.get(date)
        if target_end is None:
            raise ValueError(f"Forward-return label at {date.date()} is missing its target end date")
        target_end = pd.Timestamp(target_end)
        if target_end <= date:
            raise ValueError(f"Forward-return label at {date.date()} must end after its signal date")
        targets = pd.to_numeric(labels.loc[date].reindex(master_universe), errors="raise")
        finite_targets = np.isfinite(targets.to_numpy(dtype=float))
        target_mask = usable_blocks[end_index] & torch.tensor(finite_targets, dtype=torch.bool)
        if not target_mask.any():
            continue
        start_index = end_index - sequence_length + 1
        samples.append(torch.stack(feature_blocks[start_index : end_index + 1]))
        adjacency_samples.append(torch.stack(adjacency_blocks[start_index : end_index + 1]))
        target_samples.append(torch.tensor(targets.fillna(0.0).to_numpy(dtype=np.float32), dtype=dtype))
        mask_samples.append(target_mask)
        end_dates.append(date)
        target_end_dates.append(target_end)
    if not samples:
        raise ValueError("No trainable graph windows were produced from the supplied snapshots and labels")
    return DynamicGraphWindowDataset(
        features=torch.stack(samples),
        adjacency=torch.stack(adjacency_samples),
        target_returns=torch.stack(target_samples),
        target_mask=torch.stack(mask_samples),
        input_end_dates=tuple(end_dates),
        target_end_dates=tuple(target_end_dates),
        universe=tuple(master_universe),
        feature_columns=tuple(feature_columns),
    )


def _validate_universe(universe: Sequence[str]) -> list[str]:
    symbols = [str(symbol).strip() for symbol in universe]
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Declared master universe must contain non-empty symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Declared master universe must contain unique symbols")
    return symbols


def _snapshot_dates(
    graph_snapshots: Mapping[pd.Timestamp, nx.Graph],
    observed_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if not graph_snapshots:
        raise ValueError("At least one graph snapshot is required")
    normalized = pd.DatetimeIndex(pd.to_datetime(list(graph_snapshots), errors="raise")).sort_values().unique()
    if not normalized.isin(observed_dates).all():
        raise ValueError("Every graph snapshot date must have a corresponding node-feature/price date")
    return normalized
