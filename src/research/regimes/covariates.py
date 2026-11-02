"""Causal transformations of external public market-state price series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data import validate_price_panel


def build_price_regime_covariates(
    prices: pd.DataFrame,
    *,
    symbol: str,
    prefix: str,
    lookback_sessions: int = 21,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Derive level, return, and realised-volatility covariates from one series.

    The series is an observed market-state input, not an investable universe
    member. Every output at date ``t`` uses its close and prior closes only;
    no missing value is filled or carried forward.
    """
    if not symbol or not prefix:
        raise ValueError("symbol and prefix must be non-empty")
    if lookback_sessions < 2 or annualization_factor <= 0:
        raise ValueError("lookback_sessions must be at least two and annualization_factor positive")
    panel, _ = validate_price_panel(prices, require_ohlcv=False)
    series = panel.loc[panel["symbol"] == symbol, ["date", "close"]].copy().sort_values("date")
    if series.empty:
        raise ValueError(f"External price series does not contain {symbol!r}")
    if series["date"].duplicated().any():
        raise ValueError("External price series needs one close per date")
    returns = series["close"].pct_change(fill_method=None)
    trailing_return = series["close"].div(series["close"].shift(lookback_sessions)).sub(1.0)
    output = pd.DataFrame(
        {
            "date": series["date"].to_numpy(),
            f"{prefix}_level": series["close"].to_numpy(),
            f"{prefix}_one_session_return": returns.to_numpy(),
            f"{prefix}_trailing_return": trailing_return.to_numpy(),
            f"{prefix}_realized_volatility": returns.rolling(lookback_sessions).std(ddof=1).mul(
                np.sqrt(annualization_factor)
            ).to_numpy(),
        }
    )
    return output.dropna().reset_index(drop=True)
