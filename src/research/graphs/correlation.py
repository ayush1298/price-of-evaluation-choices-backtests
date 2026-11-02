"""Comparable correlation-network baselines with fixed vertex sets."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def build_pcc_graph(
    returns: pd.DataFrame,
    *,
    edge_fraction: float = 0.002,
    absolute_similarity: bool = True,
    min_observations: int = 60,
) -> nx.Graph:
    """Build a sparse PCC graph while retaining all input assets as vertices."""
    return _build_similarity_graph(
        _pairwise_pcc(returns, absolute_similarity=absolute_similarity, min_observations=min_observations),
        returns.columns,
        edge_fraction,
        method="pcc",
    )


def build_partial_correlation_graph(
    returns: pd.DataFrame,
    *,
    edge_fraction: float = 0.002,
    absolute_similarity: bool = True,
) -> nx.Graph:
    """Build a Ledoit-Wolf partial-correlation graph as a stronger baseline."""
    data = _validate_returns(returns).dropna(how="any")
    if len(data) < 2:
        raise ValueError("Partial correlation requires at least two complete return observations")
    estimator = LedoitWolf().fit(data.to_numpy(dtype=float))
    precision = estimator.precision_
    diagonal = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 0.0)
    pairs: list[tuple[str, str, float]] = []
    assets = data.columns.astype(str).tolist()
    for left, right in combinations(range(len(assets)), 2):
        score = float(partial[left, right])
        similarity = abs(score) if absolute_similarity else score
        if np.isfinite(similarity) and similarity > 0:
            pairs.append((assets[left], assets[right], similarity))
    return _build_similarity_graph(pairs, assets, edge_fraction, method="partial_correlation")


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("At least two asset return columns are required")
    data = returns.copy()
    data.columns = data.columns.astype(str)
    if data.columns.duplicated().any():
        raise ValueError("Return columns must be unique asset identifiers")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def _pairwise_pcc(
    returns: pd.DataFrame,
    *,
    absolute_similarity: bool,
    min_observations: int,
) -> list[tuple[str, str, float]]:
    if min_observations < 2:
        raise ValueError("min_observations must be at least two")
    data = _validate_returns(returns)
    pairs: list[tuple[str, str, float]] = []
    for left, right in combinations(data.columns, 2):
        pair = data[[left, right]].dropna()
        if len(pair) < min_observations:
            continue
        correlation = float(pair[left].corr(pair[right]))
        similarity = abs(correlation) if absolute_similarity else correlation
        if np.isfinite(similarity) and similarity > 0:
            pairs.append((left, right, similarity))
    return pairs


def _build_similarity_graph(
    pairs: list[tuple[str, str, float]],
    assets: pd.Index | list[str],
    edge_fraction: float,
    *,
    method: str,
) -> nx.Graph:
    if not 0 < edge_fraction <= 1:
        raise ValueError("edge_fraction must be in (0, 1]")
    graph = nx.Graph(method=method)
    graph.add_nodes_from(str(asset) for asset in assets)
    if not pairs:
        return graph
    edge_count = max(1, int(np.ceil(len(pairs) * edge_fraction)))
    for left, right, similarity in sorted(pairs, key=lambda row: row[2], reverse=True)[:edge_count]:
        graph.add_edge(left, right, weight=similarity)
    return graph
