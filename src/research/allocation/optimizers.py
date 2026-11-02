"""Constrained, transparent allocation baselines.

These optimisers are intentionally conventional. They provide a fair baseline
between equal-weight hub selection and a future neural graph allocator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

from .constraints import validate_long_only_weights


def inverse_volatility_weights(
    returns: pd.DataFrame,
    *,
    max_weight: float | None = None,
) -> pd.Series:
    """Create a long-only inverse-volatility allocation from historical returns."""
    data = returns.apply(pd.to_numeric, errors="raise").dropna(how="any")
    if data.empty:
        raise ValueError("Historical returns are required for inverse-volatility weights")
    volatility = data.std(ddof=1)
    if (volatility <= 0).any() or volatility.isna().any():
        raise ValueError("Every eligible asset needs positive historical volatility")
    raw = 1.0 / volatility
    weights = raw / raw.sum()
    if max_weight is not None:
        weights = _cap_and_renormalize(weights, max_weight)
    return validate_long_only_weights(weights, max_weight=max_weight)


def mean_cvar_weights(
    returns: pd.DataFrame,
    *,
    expected_returns: pd.Series | None = None,
    alpha: float = 0.05,
    cvar_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    previous_weights: pd.Series | None = None,
    max_weight: float = 1.0,
) -> pd.Series:
    """Solve a long-only fully invested mean-CVaR linear program.

    The loss is ``-expected_return + cvar_aversion * CVaR + turnover_penalty
    * one_way_turnover``. Inputs must be historical returns known at the
    signal date; the function deliberately accepts no future observations.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if cvar_aversion < 0 or turnover_penalty < 0:
        raise ValueError("Risk and turnover penalties cannot be negative")
    data = returns.apply(pd.to_numeric, errors="raise").dropna(how="any")
    if len(data) < 2:
        raise ValueError("At least two historical return observations are required")
    assets = data.columns
    n_assets, n_observations = len(assets), len(data)
    if not 1.0 / n_assets <= max_weight <= 1.0:
        raise ValueError("max_weight must permit a fully invested portfolio")

    expected = data.mean() if expected_returns is None else expected_returns.reindex(assets)
    if expected.isna().any():
        raise ValueError("Expected returns must cover every asset")
    previous = (
        pd.Series(1.0 / n_assets, index=assets)
        if previous_weights is None
        else validate_long_only_weights(previous_weights.reindex(assets, fill_value=0.0), max_weight=max_weight)
    )

    # Variables: [asset weights (n), VaR threshold eta (1), CVaR excesses (T), turnover auxiliaries (n)].
    eta_index = n_assets
    excess_start = eta_index + 1
    turnover_start = excess_start + n_observations
    n_variables = turnover_start + n_assets

    objective = np.zeros(n_variables)
    objective[:n_assets] = -expected.to_numpy(dtype=float)
    objective[eta_index] = cvar_aversion
    objective[excess_start:turnover_start] = cvar_aversion / (alpha * n_observations)
    objective[turnover_start:] = turnover_penalty

    inequalities: list[np.ndarray] = []
    bounds_vector: list[float] = []
    return_matrix = data.to_numpy(dtype=float)
    # u_t >= -r_t'w - eta  -> -r_t'w - eta - u_t <= 0
    for observation, row in enumerate(return_matrix):
        constraint = np.zeros(n_variables)
        constraint[:n_assets] = -row
        constraint[eta_index] = -1.0
        constraint[excess_start + observation] = -1.0
        inequalities.append(constraint)
        bounds_vector.append(0.0)
    # d_i >= w_i - previous_i and d_i >= previous_i - w_i
    for asset_index, previous_weight in enumerate(previous.to_numpy(dtype=float)):
        plus = np.zeros(n_variables)
        plus[asset_index] = 1.0
        plus[turnover_start + asset_index] = -1.0
        inequalities.append(plus)
        bounds_vector.append(previous_weight)

        minus = np.zeros(n_variables)
        minus[asset_index] = -1.0
        minus[turnover_start + asset_index] = -1.0
        inequalities.append(minus)
        bounds_vector.append(-previous_weight)

    equality = np.zeros((1, n_variables))
    equality[0, :n_assets] = 1.0
    result = linprog(
        c=objective,
        A_ub=np.vstack(inequalities),
        b_ub=np.array(bounds_vector),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=[(0.0, max_weight)] * n_assets + [(None, None)] + [(0.0, None)] * (n_observations + n_assets),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Mean-CVaR optimisation failed: {result.message}")
    return validate_long_only_weights(pd.Series(result.x[:n_assets], index=assets), max_weight=max_weight)


def hierarchical_risk_parity_weights(
    returns: pd.DataFrame,
    *,
    max_weight: float | None = None,
) -> pd.Series:
    """Create long-only Hierarchical Risk Parity (HRP) weights.

    HRP clusters assets using their historical correlation distance, orders the
    resulting tree, then recursively assigns capital to the less variable
    cluster.  It is entirely covariance based: unlike a mean--variance or
    mean--CVaR allocator, it does not require an unstable expected-return
    estimate.  The input is restricted to historical returns supplied by the
    caller, so this function cannot itself introduce look-ahead bias.
    """
    data = returns.apply(pd.to_numeric, errors="raise").dropna(how="any")
    if len(data) < 2:
        raise ValueError("At least two historical return observations are required")
    if data.shape[1] < 2:
        raise ValueError("Hierarchical risk parity requires at least two assets")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Historical returns must be finite")

    return hierarchical_risk_parity_weights_from_covariance(data.cov(), max_weight=max_weight)


def hierarchical_risk_parity_weights_from_covariance(
    covariance: pd.DataFrame,
    *,
    max_weight: float | None = None,
) -> pd.Series:
    """Create long-only HRP weights from a finite positive-variance covariance matrix.

    This form is used when a causal model supplies a conditional covariance
    estimate rather than a raw return panel.  It has the exact same
    clustering, recursive-bisection, cap, and validation behaviour as
    :func:`hierarchical_risk_parity_weights`.
    """
    if covariance.shape[0] < 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("HRP covariance must be a square matrix for at least two assets")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError("HRP covariance asset labels must be unique")
    if set(covariance.index) != set(covariance.columns):
        raise ValueError("HRP covariance index and columns must contain the same assets")
    matrix = covariance.reindex(index=covariance.columns, columns=covariance.columns).apply(
        pd.to_numeric, errors="raise"
    )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("HRP covariance must be finite")
    if not np.allclose(values, values.T, atol=1e-10, rtol=1e-10):
        raise ValueError("HRP covariance must be symmetric")
    variances = np.diag(values)
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("Every eligible asset needs positive historical variance")
    scales = np.sqrt(variances)
    correlation = values / np.outer(scales, scales)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    # The standard correlation distance is non-negative up to round-off.
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="single", optimal_ordering=True)
    ordered_assets = matrix.columns[leaves_list(tree)].tolist()

    weights = pd.Series(1.0, index=ordered_assets, dtype=float)
    clusters: list[list[str]] = [ordered_assets]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            midpoint = len(cluster) // 2
            left, right = cluster[:midpoint], cluster[midpoint:]
            left_variance = _cluster_variance(covariance, left)
            right_variance = _cluster_variance(covariance, right)
            allocation_to_left = 1.0 - left_variance / (left_variance + right_variance)
            weights.loc[left] *= allocation_to_left
            weights.loc[right] *= 1.0 - allocation_to_left
            next_clusters.extend([left, right])
        clusters = next_clusters
    weights = weights.reindex(matrix.columns)
    weights /= weights.sum()
    if max_weight is not None:
        weights = _cap_and_renormalize(weights, max_weight)
    return validate_long_only_weights(weights, max_weight=max_weight)


def _cluster_variance(covariance: pd.DataFrame, assets: list[str]) -> float:
    """Return the inverse-variance portfolio variance for one HRP cluster."""
    subset = covariance.loc[assets, assets]
    diagonal = np.diag(subset.to_numpy(dtype=float))
    inverse_variance = 1.0 / diagonal
    local_weights = inverse_variance / inverse_variance.sum()
    value = float(local_weights @ subset.to_numpy(dtype=float) @ local_weights)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("HRP cluster variance must be finite and positive")
    return value


def _cap_and_renormalize(weights: pd.Series, max_weight: float) -> pd.Series:
    if not 1.0 / len(weights) <= max_weight <= 1.0:
        raise ValueError("max_weight must permit a fully invested portfolio")
    capped = weights.copy()
    for _ in range(len(capped) + 1):
        over = capped > max_weight + 1e-12
        if not over.any():
            return capped / capped.sum()
        excess = float((capped[over] - max_weight).sum())
        capped.loc[over] = max_weight
        recipients = ~over
        if not recipients.any():
            break
        capacity = (max_weight - capped.loc[recipients]).clip(lower=0.0)
        if capacity.sum() <= 0:
            break
        capped.loc[recipients] += excess * capacity / capacity.sum()
    raise ValueError("Unable to satisfy max_weight cap")
