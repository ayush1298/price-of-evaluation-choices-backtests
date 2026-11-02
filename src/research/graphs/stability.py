"""Stability diagnostics for repeated graph-learning runs."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd


def edge_jaccard(left: nx.Graph, right: nx.Graph) -> float:
    """Return Jaccard overlap of undirected edge sets."""
    left_edges = {frozenset(edge) for edge in left.edges()}
    right_edges = {frozenset(edge) for edge in right.edges()}
    union = left_edges | right_edges
    return float(len(left_edges & right_edges) / len(union)) if union else 1.0


def edge_stability(graphs: Iterable[nx.Graph], seeds: Iterable[int] | None = None) -> pd.DataFrame:
    """Summarise pairwise edge overlap across independent graph runs."""
    graph_list = list(graphs)
    seed_list = list(seeds) if seeds is not None else list(range(len(graph_list)))
    if len(seed_list) != len(graph_list):
        raise ValueError("seeds and graphs must have the same length")
    rows: list[dict[str, object]] = []
    for left_index in range(len(graph_list)):
        for right_index in range(left_index + 1, len(graph_list)):
            rows.append(
                {
                    "seed_left": seed_list[left_index],
                    "seed_right": seed_list[right_index],
                    "edge_jaccard": edge_jaccard(graph_list[left_index], graph_list[right_index]),
                }
            )
    return pd.DataFrame(rows, columns=["seed_left", "seed_right", "edge_jaccard"])
