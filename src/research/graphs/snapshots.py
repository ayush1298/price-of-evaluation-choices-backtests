"""Causal rolling graph snapshots with explicit point-in-time universes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from src.research.data import eligible_universe, validate_membership_history, validate_price_panel

from .centrality import align_graph_to_universe
from .correlation import build_partial_correlation_graph, build_pcc_graph


GraphMethod = Literal["pcc", "partial_correlation"]


@dataclass(frozen=True)
class GraphSnapshotConfig:
    """Causal historical window and estimator settings for graph snapshots."""

    method: GraphMethod = "pcc"
    lookback_sessions: int = 252
    edge_fraction: float = 0.002
    minimum_fit_assets: int = 30

    def __post_init__(self) -> None:
        if self.method not in {"pcc", "partial_correlation"}:
            raise ValueError(f"Unsupported graph method: {self.method}")
        if self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be at least two")
        if not 0 < self.edge_fraction <= 1:
            raise ValueError("edge_fraction must be in (0, 1]")
        if self.minimum_fit_assets < 2:
            raise ValueError("minimum_fit_assets must be at least two")


@dataclass(frozen=True)
class GraphSnapshotResult:
    """Per-signal-date graphs and their construction audit metadata."""

    graphs: dict[pd.Timestamp, nx.Graph]
    diagnostics: pd.DataFrame


def build_graph_snapshots(
    prices: pd.DataFrame,
    universe_history: pd.DataFrame,
    signal_dates: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    config: GraphSnapshotConfig | None = None,
) -> GraphSnapshotResult:
    """Construct one graph per signal date using only then-available prices.

    Every output graph contains the complete point-in-time eligible universe.
    Assets lacking a full historical window are represented as isolates rather
    than silently removed; downstream allocators may separately decide whether
    those assets are tradable.
    """
    config = config or GraphSnapshotConfig()
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    membership = validate_membership_history(universe_history)
    matrix = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    requested = pd.DatetimeIndex(pd.to_datetime(signal_dates, errors="raise")).sort_values().unique()
    if requested.empty:
        raise ValueError("At least one signal date is required")
    observed_dates = pd.DatetimeIndex(matrix.index)
    if not requested.isin(observed_dates).all():
        missing = requested[~requested.isin(observed_dates)]
        raise ValueError(f"Signal dates are not observed trading dates: {missing[:3].tolist()}")

    graphs: dict[pd.Timestamp, nx.Graph] = {}
    diagnostics: list[dict[str, object]] = []
    for date in requested:
        eligible = eligible_universe(membership, date)
        history = matrix.loc[matrix.index <= date].reindex(columns=eligible)
        window = history.tail(config.lookback_sessions + 1)
        fit_assets = window.columns[window.notna().all()].tolist()
        if len(fit_assets) < config.minimum_fit_assets:
            raise ValueError(
                f"Only {len(fit_assets)} eligible assets have a complete {config.lookback_sessions}-session "
                f"history at {date.date()}; need {config.minimum_fit_assets}"
            )
        returns = window.loc[:, fit_assets].pct_change(fill_method=None).iloc[1:]
        if config.method == "pcc":
            graph = build_pcc_graph(
                returns,
                edge_fraction=config.edge_fraction,
                min_observations=config.lookback_sessions,
            )
        else:
            graph = build_partial_correlation_graph(returns, edge_fraction=config.edge_fraction)
        graph = align_graph_to_universe(graph, eligible)
        graph.graph.update(
            {
                "signal_date": date.isoformat(),
                "method": config.method,
                "lookback_sessions": config.lookback_sessions,
                "n_eligible": len(eligible),
                "n_fit_assets": len(fit_assets),
            }
        )
        graphs[pd.Timestamp(date)] = graph
        diagnostics.append(
            {
                "signal_date": pd.Timestamp(date),
                "method": config.method,
                "lookback_sessions": config.lookback_sessions,
                "n_eligible": len(eligible),
                "n_fit_assets": len(fit_assets),
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "graph_density": nx.density(graph),
            }
        )
    return GraphSnapshotResult(graphs=graphs, diagnostics=pd.DataFrame(diagnostics))


def adjacency_matrix(graph: nx.Graph, universe: list[str]) -> np.ndarray:
    """Export a symmetric non-negative similarity matrix in a declared order."""
    aligned = align_graph_to_universe(graph, universe)
    matrix = nx.to_numpy_array(aligned, nodelist=universe, weight="weight", dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Graph adjacency must contain finite non-negative similarity weights")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Dynamic graph model currently requires undirected symmetric adjacency matrices")
    return matrix
