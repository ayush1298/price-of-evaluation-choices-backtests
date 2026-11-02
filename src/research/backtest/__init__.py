"""Daily, cost-aware, point-in-time portfolio backtesting."""

from .engine import BacktestConfig, BacktestResult, run_backtest
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    month_end_signal_dates,
    rebalance_signal_dates,
    run_walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "month_end_signal_dates",
    "rebalance_signal_dates",
    "run_backtest",
    "run_walk_forward",
]
