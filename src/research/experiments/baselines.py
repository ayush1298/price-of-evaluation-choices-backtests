"""Conventional graph and allocation baselines for the canonical pipeline.

The builder deliberately separates four choices that were entangled in the
legacy scripts: point-in-time universe, graph estimator, graph selector, and
portfolio allocator.  This makes fair ablations possible without rewriting the
walk-forward/backtest machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from src.research.allocation import hierarchical_risk_parity_weights, inverse_volatility_weights, mean_cvar_weights
from src.research.allocation.constraints import validate_long_only_weights
from src.research.data import eligible_universe, validate_membership_history
from src.research.graphs import build_partial_correlation_graph, build_pcc_graph, compute_centrality_scores


GraphMethod = Literal["pcc", "partial_correlation"]
Selector = Literal["all", "hub_avoid", "hub_follow"]
Allocator = Literal["equal_weight", "inverse_volatility", "hierarchical_risk_parity", "mean_cvar"]


@dataclass(frozen=True)
class BaselineStrategyConfig:
    """The independently configurable stages of a conventional baseline."""

    graph_method: GraphMethod = "pcc"
    selector: Selector = "all"
    allocator: Allocator = "equal_weight"
    lookback_sessions: int = 252
    minimum_assets: int = 30
    portfolio_size: int | None = 30
    edge_fraction: float = 0.002
    max_weight: float = 0.05
    cvar_alpha: float = 0.05
    cvar_aversion: float = 1.0
    turnover_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.graph_method not in {"pcc", "partial_correlation"}:
            raise ValueError(f"Unsupported graph method: {self.graph_method}")
        if self.selector not in {"all", "hub_avoid", "hub_follow"}:
            raise ValueError(f"Unsupported selector: {self.selector}")
        if self.allocator not in {"equal_weight", "inverse_volatility", "hierarchical_risk_parity", "mean_cvar"}:
            raise ValueError(f"Unsupported allocator: {self.allocator}")
        if self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be at least two")
        if self.minimum_assets < 2:
            raise ValueError("minimum_assets must be at least two")
        if self.portfolio_size is not None and self.portfolio_size < self.minimum_assets:
            raise ValueError("portfolio_size cannot be smaller than minimum_assets")
        if not 0 < self.edge_fraction <= 1:
            raise ValueError("edge_fraction must be in (0, 1]")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.portfolio_size is not None and self.portfolio_size * self.max_weight < 1 - 1e-12:
            raise ValueError("portfolio_size and max_weight cannot form a fully invested portfolio")


@dataclass
class BaselineSignalBuilder:
    """Generate a target using only prices and universe state known at the signal date.

    ``diagnostics`` records the universe, graph, and selection at every
    rebalance. It is designed to be saved with a canonical run so a reported
    result can be audited without serialising a large graph object per month.
    """

    config: BaselineStrategyConfig
    universe_history: pd.DataFrame
    diagnostics: list[dict[str, object]] = field(default_factory=list, init=False)
    _previous_weights: pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.universe_history = validate_membership_history(self.universe_history)

    def __call__(self, history: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
        eligible = eligible_universe(self.universe_history, signal_date)
        returns = self._historical_returns(history, eligible)
        graph = self._build_graph(returns)
        selected, scores = self._select_assets(returns.columns.tolist(), graph)
        weights = self._allocate(returns.loc[:, selected])
        self._previous_weights = weights.copy()
        self.diagnostics.append(self._diagnostic_row(signal_date, eligible, returns, graph, selected, scores))
        return weights

    def diagnostics_frame(self) -> pd.DataFrame:
        """Return one audit row per generated target portfolio."""
        return pd.DataFrame(self.diagnostics)

    def _historical_returns(self, history: pd.DataFrame, eligible: list[str]) -> pd.DataFrame:
        required = {"date", "symbol", "close"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"Price history is missing columns: {sorted(missing)}")
        prices = history.loc[:, ["date", "symbol", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        prices["symbol"] = prices["symbol"].astype(str)
        prices["close"] = pd.to_numeric(prices["close"], errors="raise")
        matrix = prices.pivot(index="date", columns="symbol", values="close").sort_index()
        candidates = sorted(set(eligible).intersection(matrix.columns.astype(str)))
        if len(candidates) < self.config.minimum_assets:
            raise ValueError(
                f"Only {len(candidates)} eligible assets have observed prices; "
                f"need at least {self.config.minimum_assets}"
            )
        window = matrix.reindex(columns=candidates).tail(self.config.lookback_sessions + 1)
        complete_assets = window.columns[window.notna().all()].tolist()
        if len(complete_assets) < self.config.minimum_assets:
            raise ValueError(
                "Insufficient continuous point-in-time price history for the eligible universe: "
                f"found {len(complete_assets)}, need {self.config.minimum_assets}"
            )
        returns = window.loc[:, complete_assets].pct_change(fill_method=None).iloc[1:]
        if len(returns) < self.config.lookback_sessions:
            raise ValueError(
                f"Only {len(returns)} return observations are available; need {self.config.lookback_sessions}"
            )
        if not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise ValueError("Historical return window contains non-finite values")
        return returns

    def _build_graph(self, returns: pd.DataFrame) -> nx.Graph:
        if self.config.graph_method == "pcc":
            return build_pcc_graph(
                returns,
                edge_fraction=self.config.edge_fraction,
                min_observations=self.config.lookback_sessions,
            )
        return build_partial_correlation_graph(returns, edge_fraction=self.config.edge_fraction)

    def _select_assets(self, universe: list[str], graph: nx.Graph) -> tuple[list[str], pd.DataFrame]:
        scores = compute_centrality_scores(graph, universe=universe)
        if self.config.selector == "all":
            return sorted(universe), scores
        if self.config.portfolio_size is None:
            raise ValueError("hub selectors require a finite portfolio_size")
        if len(universe) < self.config.portfolio_size:
            raise ValueError(
                f"Only {len(universe)} assets have complete data; need portfolio_size={self.config.portfolio_size}"
            )
        ascending = self.config.selector == "hub_avoid"
        ranked = scores.sort_values(
            ["composite_score", "ticker"], ascending=[ascending, True], kind="stable"
        )
        return ranked.head(self.config.portfolio_size)["ticker"].tolist(), scores

    def _allocate(self, selected_returns: pd.DataFrame) -> pd.Series:
        if self.config.allocator == "equal_weight":
            weights = pd.Series(1.0 / selected_returns.shape[1], index=selected_returns.columns)
            return validate_long_only_weights(weights, max_weight=self.config.max_weight)
        if self.config.allocator == "inverse_volatility":
            return inverse_volatility_weights(selected_returns, max_weight=self.config.max_weight)
        if self.config.allocator == "hierarchical_risk_parity":
            return hierarchical_risk_parity_weights(selected_returns, max_weight=self.config.max_weight)
        previous = None
        if self._previous_weights is not None:
            previous = self._previous_weights.reindex(selected_returns.columns, fill_value=0.0)
        return mean_cvar_weights(
            selected_returns,
            alpha=self.config.cvar_alpha,
            cvar_aversion=self.config.cvar_aversion,
            turnover_penalty=self.config.turnover_penalty,
            previous_weights=previous,
            max_weight=self.config.max_weight,
        )

    def _diagnostic_row(
        self,
        signal_date: pd.Timestamp,
        eligible: list[str],
        returns: pd.DataFrame,
        graph: nx.Graph,
        selected: list[str],
        scores: pd.DataFrame,
    ) -> dict[str, object]:
        centrality = scores.set_index("ticker").reindex(selected)["composite_score"]
        return {
            "signal_date": pd.Timestamp(signal_date),
            "graph_method": self.config.graph_method,
            "selector": self.config.selector,
            "allocator": self.config.allocator,
            "n_eligible": len(eligible),
            "n_with_complete_history": returns.shape[1],
            "n_selected": len(selected),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "graph_density": nx.density(graph),
            "selected_mean_composite_centrality": float(centrality.mean()),
            "selected_symbols": ",".join(selected),
        }
