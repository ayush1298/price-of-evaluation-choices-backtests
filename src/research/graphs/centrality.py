"""Centrality calculations with financial-network weight semantics."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd


def align_graph_to_universe(graph: nx.Graph, universe: Iterable[str]) -> nx.Graph:
    """Return a copy of ``graph`` containing all eligible assets as nodes.

    Isolated assets carry useful information in a sparse financial network.
    Excluding them changes density and makes hub-avoid portfolios incomparable
    across graph-construction methods.
    """
    aligned = graph.copy()
    aligned.add_nodes_from(str(symbol) for symbol in universe)
    return aligned


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= np.finfo(float).eps:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _distance_graph(graph: nx.Graph, similarity_weight: str) -> nx.Graph:
    """Create a graph with inverse-similarity distances for shortest paths."""
    distance_graph = graph.copy()
    for _, _, data in distance_graph.edges(data=True):
        similarity = float(data.get(similarity_weight, 1.0))
        if not np.isfinite(similarity) or similarity <= 0:
            raise ValueError(
                f"Similarity weights must be finite and positive; got {similarity!r}"
            )
        data["_distance"] = 1.0 / similarity
    return distance_graph


def compute_centrality_scores(
    graph: nx.Graph,
    *,
    universe: Iterable[str] | None = None,
    similarity_weight: str = "weight",
) -> pd.DataFrame:
    """Compute per-graph centrality scores with explicit weight handling.

    Edge weights represent co-investment *similarity*. Eigenvector and strength
    centrality use that similarity directly, while betweenness uses inverse
    similarity as a path distance. Scores are standardised within this graph,
    never across years.
    """
    if graph.is_directed():
        raise ValueError("Centrality scores currently require an undirected graph")
    if universe is not None:
        graph = align_graph_to_universe(graph, universe)
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "ticker",
                "degree_centrality",
                "strength_centrality",
                "betweenness_centrality",
                "eigenvector_centrality",
                "composite_score",
            ]
        )

    nodes = list(graph.nodes())
    degree = nx.degree_centrality(graph)
    strength = dict(graph.degree(weight=similarity_weight))
    distance_graph = _distance_graph(graph, similarity_weight)
    betweenness = nx.betweenness_centrality(distance_graph, weight="_distance")

    try:
        eigenvector = nx.eigenvector_centrality(graph, weight=similarity_weight, max_iter=2_000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0.0 for node in nodes}

    scores = pd.DataFrame(
        {
            "ticker": nodes,
            "degree_centrality": [degree[node] for node in nodes],
            "strength_centrality": [strength[node] for node in nodes],
            "betweenness_centrality": [betweenness[node] for node in nodes],
            "eigenvector_centrality": [eigenvector[node] for node in nodes],
        }
    )
    components = [
        "degree_centrality",
        "strength_centrality",
        "betweenness_centrality",
        "eigenvector_centrality",
    ]
    scores["composite_score"] = sum(_zscore(scores[column]) for column in components) / len(components)
    return scores.sort_values("composite_score", ascending=False, kind="stable").reset_index(drop=True)
