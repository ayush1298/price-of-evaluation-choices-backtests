"""Performance metrics computed from daily net returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    daily_returns: pd.Series,
    *,
    annualization_factor: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate daily-NAV performance metrics without annual sampling bias."""
    returns = pd.to_numeric(daily_returns, errors="raise").dropna().astype(float)
    if returns.empty:
        raise ValueError("At least one daily return is required")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    risk_free_daily = (1.0 + annual_risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess = returns - risk_free_daily
    volatility = returns.std(ddof=1)
    annualized_volatility = float(volatility * np.sqrt(annualization_factor)) if len(returns) > 1 else 0.0
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(annualization_factor)) if excess.std(ddof=1) > 0 else 0.0

    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(excess.mean() / downside_deviation * np.sqrt(annualization_factor)) if downside_deviation > 0 else 0.0

    wealth = (1.0 + returns).cumprod()
    wealth_with_initial = pd.concat([pd.Series([1.0]), wealth], ignore_index=True)
    max_drawdown = float((wealth_with_initial / wealth_with_initial.cummax() - 1.0).min())
    annualized_return = float(wealth.iloc[-1] ** (annualization_factor / len(returns)) - 1.0)
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    cvar_95 = float(returns[returns <= returns.quantile(0.05)].mean())

    return {
        "total_return": float(wealth.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "cvar_95_daily": cvar_95,
        "n_observations": float(len(returns)),
    }
