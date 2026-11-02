"""Walk-forward orchestration for leakage-safe signal generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_backtest


SignalBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class WalkForwardConfig:
    minimum_history_sessions: int = 252
    rebalance_frequency: str = "ME"
    first_signal_date: str | None = None
    last_signal_date: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_history_sessions < 2:
            raise ValueError("minimum_history_sessions must be at least two")
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
class WalkForwardResult:
    targets: pd.DataFrame
    backtest: BacktestResult


def rebalance_signal_dates(prices: pd.DataFrame, frequency: str = "ME") -> pd.DatetimeIndex:
    """Return the last observed market date in each requested rebalance period."""
    dates = pd.DatetimeIndex(pd.to_datetime(prices["date"], errors="raise").unique()).sort_values()
    if dates.empty:
        raise ValueError("Cannot derive signal dates from empty prices")
    try:
        signal_dates = pd.Series(dates, index=dates).resample(frequency).max().dropna()
    except ValueError as error:
        raise ValueError(f"Invalid rebalance frequency {frequency!r}") from error
    return pd.DatetimeIndex(signal_dates.to_numpy())


def month_end_signal_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the last observed market date in each calendar month.

    Kept as an explicit compatibility helper; new callers should use
    :func:`rebalance_signal_dates` so configuration is honored.
    """
    return rebalance_signal_dates(prices, "ME")


def run_walk_forward(
    prices: pd.DataFrame,
    signal_builder: SignalBuilder,
    *,
    walk_forward_config: WalkForwardConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Build target weights using only past/current data, then backtest them.

    ``signal_builder`` receives all price observations on or before the signal
    date. It never receives a future return or future price row.
    """
    walk_forward_config = walk_forward_config or WalkForwardConfig()
    working = prices.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise")
    trading_dates = pd.DatetimeIndex(working["date"].unique()).sort_values()
    signal_dates = rebalance_signal_dates(working, walk_forward_config.rebalance_frequency)
    if walk_forward_config.first_signal_date is not None:
        first_signal_date = pd.Timestamp(walk_forward_config.first_signal_date)
        signal_dates = signal_dates[signal_dates >= first_signal_date]
    if walk_forward_config.last_signal_date is not None:
        last_signal_date = pd.Timestamp(walk_forward_config.last_signal_date)
        signal_dates = signal_dates[signal_dates <= last_signal_date]
    target_rows: list[dict[str, object]] = []

    for signal_date in signal_dates:
        history_dates = trading_dates[trading_dates <= signal_date]
        if len(history_dates) < walk_forward_config.minimum_history_sessions:
            continue
        history = working.loc[working["date"] <= signal_date].copy()
        weights = signal_builder(history, pd.Timestamp(signal_date))
        if not isinstance(weights, pd.Series):
            raise TypeError("signal_builder must return a pandas Series indexed by symbol")
        for symbol, weight in weights.items():
            target_rows.append(
                {"signal_date": pd.Timestamp(signal_date), "symbol": str(symbol), "weight": float(weight)}
            )

    targets = pd.DataFrame(target_rows, columns=["signal_date", "symbol", "weight"])
    if targets.empty:
        raise ValueError("No targets generated; increase history coverage or lower minimum_history_sessions")
    evaluation_prices = working
    if walk_forward_config.last_signal_date is not None:
        evaluation_prices = working.loc[
            working["date"] <= pd.Timestamp(walk_forward_config.last_signal_date)
        ].copy()
    return WalkForwardResult(targets=targets, backtest=run_backtest(evaluation_prices, targets, config=backtest_config))
