"""A small, explicit daily portfolio backtest engine.

Signals are dated at the information close. With close-only inputs, a target
executes at the configured future *close*, after the portfolio has earned that
session's return under its previous weights. The new target first earns a
return on the following session. All costs are charged in the execution ledger.
The engine deliberately fails on missing held prices rather than silently
dropping a holding from a portfolio return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.allocation.constraints import validate_long_only_weights
from src.research.data.schema import validate_price_panel


@dataclass(frozen=True)
class BacktestConfig:
    execution_lag_sessions: int = 1
    transaction_cost_bps: float = 10.0
    initial_nav: float = 1.0
    max_weight: float | None = None
    min_rebalance_turnover: float = 0.0

    def __post_init__(self) -> None:
        if self.execution_lag_sessions < 1:
            raise ValueError("execution_lag_sessions must be at least one")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        if not 0.0 <= self.min_rebalance_turnover <= 1.0:
            raise ValueError("min_rebalance_turnover must be in [0, 1]")


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    holdings: pd.DataFrame
    executions: pd.DataFrame


def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    validated, _ = validate_price_panel(prices, require_ohlcv=False)
    matrix = validated.pivot(index="date", columns="symbol", values="close").sort_index()
    if matrix.index.has_duplicates or not matrix.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and ordered")
    return matrix


def _scheduled_targets(
    targets: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    config: BacktestConfig,
) -> dict[pd.Timestamp, pd.Series]:
    required = {"signal_date", "symbol", "weight"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Target weights are missing columns: {sorted(missing)}")

    schedule = targets.loc[:, ["signal_date", "symbol", "weight"]].copy()
    schedule["signal_date"] = pd.to_datetime(schedule["signal_date"], errors="raise")
    schedule["symbol"] = schedule["symbol"].astype(str)
    if schedule.duplicated(["signal_date", "symbol"]).any():
        raise ValueError("Each target may contain one weight per signal date and symbol")

    executions: dict[pd.Timestamp, pd.Series] = {}
    for signal_date, group in schedule.groupby("signal_date", sort=True):
        position = int(trading_dates.searchsorted(signal_date, side="right"))
        position += config.execution_lag_sessions - 1
        if position >= len(trading_dates):
            continue
        execution_date = trading_dates[position]
        if execution_date in executions:
            raise ValueError(f"Multiple target portfolios execute on {execution_date.date()}")
        target = group.set_index("symbol")["weight"]
        executions[execution_date] = validate_long_only_weights(target, max_weight=config.max_weight)
    return executions


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest long-only signal-date targets against daily close-to-close returns.

    ``targets`` contains the desired allocation after each signal close. With
    close-price data, the target is implemented at the next tradable *close*
    by default, so it first participates in the next session's close-to-close
    return. Residual weight is held as zero-return cash. Portfolio weights
    drift between rebalances, so turnover reflects the actual pre-trade
    portfolio after that session's return.
    """
    config = config or BacktestConfig()
    matrix = _price_matrix(prices)
    trading_dates = pd.DatetimeIndex(matrix.index)
    scheduled = _scheduled_targets(targets, trading_dates, config)
    if not scheduled:
        raise ValueError("No target portfolio can be executed on the available trading dates")
    first_execution_date = min(scheduled)
    returns = matrix.pct_change(fill_method=None)

    all_assets = matrix.columns.astype(str)
    weights = pd.Series(0.0, index=all_assets, dtype=float)
    cash_weight = 1.0
    nav = float(config.initial_nav)
    daily_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []

    for row_index, date in enumerate(trading_dates):
        # Training/warm-up cash before a strategy's first executable target is
        # not investable performance. Omitting it prevents annualised metrics
        # from being diluted by an arbitrary modelling-history length.
        if date < first_execution_date:
            continue
        transaction_cost = 0.0
        turnover = 0.0
        requested_turnover = 0.0
        rebalanced = False
        skipped_by_no_trade_band = False
        if row_index == 0:
            asset_returns = pd.Series(0.0, index=all_assets)
        else:
            asset_returns = returns.loc[date]
            held = weights[weights.abs() > 1e-14].index
            if asset_returns.loc[held].isna().any():
                unavailable = asset_returns.loc[held][asset_returns.loc[held].isna()].index.tolist()
                raise ValueError(
                    f"Held assets have missing prices on {date.date()}: {unavailable}. "
                    "Fix the point-in-time universe or explicitly liquidate them before this date."
                )
            asset_returns = asset_returns.fillna(0.0)

        gross_return = float((weights * asset_returns).sum())
        gross_growth = 1.0 + gross_return
        if gross_growth <= 0:
            raise ValueError(f"Portfolio lost 100% or more on {date.date()}")
        weights = weights * (1.0 + asset_returns) / gross_growth
        cash_weight = cash_weight / gross_growth

        # At the future close, first account for the old portfolio's daily
        # return. Then trade into the target and charge the implementation
        # cost. This avoids giving a post-close signal the next close's full
        # close-to-close move for free.
        if date in scheduled:
            target = scheduled[date].reindex(all_assets, fill_value=0.0)
            current_with_cash = pd.concat([weights, pd.Series({"__cash__": cash_weight})])
            target_cash = float(1.0 - target.sum())
            target_with_cash = pd.concat([target, pd.Series({"__cash__": target_cash})])
            requested_turnover = float(0.5 * (target_with_cash - current_with_cash).abs().sum())
            skipped_by_no_trade_band = requested_turnover < config.min_rebalance_turnover
            if not skipped_by_no_trade_band:
                turnover = requested_turnover
                transaction_cost = turnover * config.transaction_cost_bps / 10_000.0
                weights = target
                cash_weight = target_cash
                rebalanced = True
            execution_rows.append(
                {
                    "execution_date": date,
                    "requested_turnover": requested_turnover,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "target_cash_weight": target_cash,
                    "n_positions": int((weights > 0).sum()),
                    "skipped_by_no_trade_band": skipped_by_no_trade_band,
                }
            )

        net_return = float((1.0 + gross_return) * (1.0 - transaction_cost) - 1.0)
        nav *= 1.0 + net_return

        daily_rows.append(
            {
                "date": date,
                "nav": nav,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "requested_turnover": requested_turnover,
                "turnover": turnover,
                "rebalanced": rebalanced,
                "skipped_by_no_trade_band": skipped_by_no_trade_band,
                "cash_weight": cash_weight,
            }
        )
        for symbol, weight in weights[weights.abs() > 1e-14].items():
            holdings_rows.append({"date": date, "symbol": symbol, "weight": float(weight)})

    daily = pd.DataFrame(daily_rows).set_index("date")
    # Include the initial pre-trade NAV so an initial transaction cost or loss
    # is recognised as drawdown rather than becoming the artificial first peak.
    peak = daily["nav"].cummax().clip(lower=config.initial_nav)
    daily["drawdown"] = daily["nav"].div(peak).sub(1.0)
    holdings = pd.DataFrame(holdings_rows, columns=["date", "symbol", "weight"])
    executions = pd.DataFrame(
        execution_rows,
        columns=[
            "execution_date",
            "requested_turnover",
            "turnover",
            "transaction_cost",
            "target_cash_weight",
            "n_positions",
            "skipped_by_no_trade_band",
        ],
    )
    return BacktestResult(daily=daily, holdings=holdings, executions=executions)
