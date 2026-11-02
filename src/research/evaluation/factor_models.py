"""Date-aligned factor-regression diagnostics for canonical portfolio returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRegressionResult:
    """OLS factor model with Newey-West standard errors for daily alpha."""

    observations: int
    factor_columns: tuple[str, ...]
    risk_free_column: str
    hac_lags: int
    daily_alpha: float
    annualized_alpha_linear: float
    alpha_hac_standard_error: float
    alpha_hac_t_statistic: float
    alpha_hac_two_sided_p_value: float
    r_squared: float
    factor_betas: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_factor_model(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    risk_free_column: str = "RF",
    annualization_factor: int = 252,
    hac_lags: int = 5,
) -> FactorRegressionResult:
    """Fit excess portfolio returns to daily factors using HAC inference.

    ``portfolio_returns`` and factor values must be decimal daily returns and
    have date indexes/columns that can be converted to trading dates.  The
    alpha annualisation is the conventional linear daily-alpha times 252; it
    is labelled explicitly to avoid presenting it as a compounded return.
    """
    if annualization_factor < 1 or hac_lags < 0:
        raise ValueError("annualization_factor must be positive and hac_lags non-negative")
    selected = tuple(factor_columns)
    if not selected:
        raise ValueError("At least one factor column is required")
    required = {"date", risk_free_column, *selected}
    missing = required.difference(factors.columns)
    if missing:
        raise ValueError(f"Factor input is missing required columns: {sorted(missing)}")

    returns = pd.to_numeric(portfolio_returns, errors="raise").rename("portfolio_return")
    returns.index = pd.to_datetime(returns.index, errors="raise")
    if returns.index.has_duplicates:
        raise ValueError("Portfolio returns must have unique dates")
    factor_frame = factors.loc[:, ["date", *selected, risk_free_column]].copy()
    factor_frame["date"] = pd.to_datetime(factor_frame["date"], errors="raise")
    factor_frame.loc[:, [*selected, risk_free_column]] = factor_frame.loc[
        :, [*selected, risk_free_column]
    ].apply(pd.to_numeric, errors="raise")
    if factor_frame["date"].duplicated().any():
        raise ValueError("Factor dates must be unique")
    aligned = pd.concat([returns, factor_frame.set_index("date")], axis=1, join="inner").dropna()
    if len(aligned) <= len(selected) + 2:
        raise ValueError("Not enough aligned observations for the requested factor model")
    values = aligned.loc[:, ["portfolio_return", *selected, risk_free_column]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Portfolio and factor returns must be finite")

    response = values[:, 0] - values[:, -1]
    design = np.column_stack([np.ones(len(aligned)), values[:, 1:-1]])
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    covariance = _newey_west_covariance(design, residuals, hac_lags)
    alpha_standard_error = float(sqrt(max(0.0, covariance[0, 0])))
    alpha_t = float(coefficients[0] / alpha_standard_error) if alpha_standard_error > 0 else 0.0
    alpha_p = float(erfc(abs(alpha_t) / sqrt(2.0)))
    total_sum_squares = float(np.square(response - response.mean()).sum())
    r_squared = 1.0 - float(np.square(residuals).sum()) / total_sum_squares if total_sum_squares > 0 else 0.0
    return FactorRegressionResult(
        observations=len(aligned),
        factor_columns=selected,
        risk_free_column=risk_free_column,
        hac_lags=hac_lags,
        daily_alpha=float(coefficients[0]),
        annualized_alpha_linear=float(coefficients[0] * annualization_factor),
        alpha_hac_standard_error=alpha_standard_error,
        alpha_hac_t_statistic=alpha_t,
        alpha_hac_two_sided_p_value=alpha_p,
        r_squared=float(r_squared),
        factor_betas={column: float(value) for column, value in zip(selected, coefficients[1:])},
    )


def _newey_west_covariance(design: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """Return the unscaled Newey-West covariance estimator for OLS coefficients."""
    observations = len(design)
    effective_lags = min(lags, observations - 1)
    score = design * residuals[:, None]
    meat = score.T @ score
    for lag in range(1, effective_lags + 1):
        weight = 1.0 - lag / (effective_lags + 1.0)
        autocovariance = score[lag:].T @ score[:-lag]
        meat += weight * (autocovariance + autocovariance.T)
    inverse_gram = np.linalg.pinv(design.T @ design)
    return inverse_gram @ meat @ inverse_gram
