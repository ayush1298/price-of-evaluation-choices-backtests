"""Causal market and cross-sectional features for regime models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data import validate_price_panel


def build_market_regime_features(
    prices: pd.DataFrame,
    *,
    market_symbol: str | None = None,
    volatility_window: int = 21,
    drawdown_window: int = 63,
    annualization_factor: int = 252,
    minimum_cross_sectional_assets: int = 10,
    external_covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build daily features known at each market close.

    With ``market_symbol`` the market return is that benchmark's realised
    return. Without one, it is the contemporaneous equal-weight return across
    the observed cross section; this fallback is explicit so it cannot be
    mistaken for an external market index. No fill-forward price is applied.
    """
    if volatility_window < 2 or drawdown_window < 2:
        raise ValueError("volatility_window and drawdown_window must be at least two")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if minimum_cross_sectional_assets < 2:
        raise ValueError("minimum_cross_sectional_assets must be at least two")
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    matrix = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = matrix.pct_change(fill_method=None)
    if market_symbol is not None:
        if market_symbol not in returns.columns:
            raise ValueError(f"market_symbol {market_symbol!r} is absent from the price panel")
        market_returns = returns[market_symbol]
        source = "benchmark"
    else:
        observed_count = returns.notna().sum(axis=1)
        market_returns = returns.mean(axis=1, skipna=True).where(observed_count >= minimum_cross_sectional_assets)
        source = "cross_section_equal_weight"

    wealth = (1.0 + market_returns).cumprod()
    trailing_return = (1.0 + market_returns).rolling(volatility_window).apply(np.prod, raw=True).sub(1.0)
    features = pd.DataFrame(
        {
            "date": matrix.index,
            "market_return": market_returns,
            "realized_volatility": market_returns.rolling(volatility_window).std(ddof=1)
            * np.sqrt(annualization_factor),
            "trailing_market_return": trailing_return,
            "market_drawdown": wealth.div(wealth.rolling(drawdown_window).max()).sub(1.0),
            "cross_sectional_dispersion": returns.std(axis=1, ddof=1),
            "cross_sectional_assets": returns.notna().sum(axis=1),
            "market_source": source,
        }
    )
    output = features.dropna(
        subset=[
            "market_return",
            "realized_volatility",
            "trailing_market_return",
            "market_drawdown",
            "cross_sectional_dispersion",
        ]
    ).reset_index(drop=True)
    if external_covariates is not None:
        output = merge_external_regime_covariates(output, external_covariates)
    return output


def merge_external_regime_covariates(
    market_features: pd.DataFrame,
    external_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Exact-date merge finite external covariates without forward filling."""
    if "date" not in market_features.columns or "date" not in external_covariates.columns:
        raise ValueError("Market features and external covariates each require a date column")
    covariates = external_covariates.copy()
    columns = [column for column in covariates.columns if column != "date"]
    if not columns:
        raise ValueError("External covariates need at least one numeric column")
    overlap = set(columns).intersection(market_features.columns)
    if overlap:
        raise ValueError(f"External covariates overlap market feature columns: {sorted(overlap)}")
    covariates["date"] = pd.to_datetime(covariates["date"], errors="raise")
    if covariates["date"].duplicated().any():
        raise ValueError("External covariates need one row per date")
    for column in columns:
        covariates[column] = pd.to_numeric(covariates[column], errors="raise")
    values = covariates.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("External covariates must be finite")
    features = market_features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    return features.merge(covariates, on="date", how="inner", validate="one_to_one")


def merge_graph_regime_features(
    market_features: pd.DataFrame,
    graph_diagnostics: pd.DataFrame,
    *,
    graph_date_column: str = "signal_date",
) -> pd.DataFrame:
    """Join independently generated signal-date graph diagnostics to market features.

    The caller is responsible for constructing the diagnostic graph from
    information available at that same signal close. The merge performs an
    exact-date inner join rather than filling graph features across dates.
    """
    required_market = {"date"}
    required_graph = {graph_date_column, "graph_density", "graph_edges"}
    if missing := required_market.difference(market_features.columns):
        raise ValueError(f"Market regime features are missing columns: {sorted(missing)}")
    if missing := required_graph.difference(graph_diagnostics.columns):
        raise ValueError(f"Graph diagnostics are missing columns: {sorted(missing)}")
    market = market_features.copy()
    graph = graph_diagnostics.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    graph[graph_date_column] = pd.to_datetime(graph[graph_date_column], errors="raise")
    if market["date"].duplicated().any() or graph[graph_date_column].duplicated().any():
        raise ValueError("Regime feature inputs must have at most one row per date")
    graph = graph.rename(columns={graph_date_column: "date"})
    return market.merge(graph, on="date", how="inner", validate="one_to_one")


def broadcast_regime_probabilities_to_nodes(
    node_features: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    *,
    date_column: str = "date",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Exact-date join causal regime probabilities onto every node feature row.

    No probability is forward-filled: snapshots without a probability are
    discarded.  This makes the downstream model unavailable until its regime
    encoder has accumulated enough earlier market observations, which is the
    intended causal warm-up behaviour.
    """
    required_nodes = {date_column, "symbol"}
    if missing := required_nodes.difference(node_features.columns):
        raise ValueError(f"Node features are missing columns: {sorted(missing)}")
    if date_column not in regime_probabilities.columns:
        raise ValueError("Regime probabilities are missing the date column")
    probability_columns = tuple(
        column for column in regime_probabilities.columns if column.startswith("probability_state_")
    )
    if not probability_columns:
        raise ValueError("Regime probabilities need one or more probability_state_* columns")
    nodes = node_features.copy()
    regimes = regime_probabilities.loc[:, [date_column, *probability_columns]].copy()
    nodes[date_column] = pd.to_datetime(nodes[date_column], errors="raise")
    regimes[date_column] = pd.to_datetime(regimes[date_column], errors="raise")
    if regimes[date_column].duplicated().any():
        raise ValueError("Regime probabilities must have one row per date")
    for column in probability_columns:
        regimes[column] = pd.to_numeric(regimes[column], errors="raise")
    probability_values = regimes.loc[:, probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probability_values).all() or (probability_values < 0).any():
        raise ValueError("Regime probabilities must be finite and non-negative")
    if not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Regime probabilities must sum to one on every date")
    merged = nodes.merge(regimes, on=date_column, how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError("No node feature dates overlap causal regime probabilities")
    return merged, probability_columns
