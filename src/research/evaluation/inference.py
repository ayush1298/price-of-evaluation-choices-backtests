"""Paired out-of-sample inference for canonical portfolio comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import e
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlockBootstrapComparison:
    """Paired moving-block bootstrap evidence for a return differential."""

    observations: int
    block_length: int
    replications: int
    annualization_factor: int
    mean_daily_excess_return: float
    annualized_excess_return: float
    observed_annualized_sharpe_difference: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    one_sided_p_value: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated-Sharpe diagnostic conditional on the tested model family."""

    observations: int
    trials: int
    observed_periodic_sharpe: float
    expected_max_trial_sharpe: float
    deflated_sharpe_probability: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class WhiteRealityCheckResult:
    """Moving-block White Reality Check for a searched candidate family."""

    observations: int
    candidates: int
    block_length: int
    replications: int
    annualization_factor: int
    best_candidate: str
    best_mean_daily_excess_return: float
    best_annualized_excess_return: float
    reality_check_p_value: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def paired_moving_block_bootstrap(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    block_length: int = 21,
    replications: int = 2_000,
    annualization_factor: int = 252,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> BlockBootstrapComparison:
    """Test an aligned net-return differential while preserving serial dependence.

    The returned p-value tests the one-sided null that mean strategy excess
    return is non-positive. Inputs are paired by their date index, so a market
    event affects both strategies in the same sampled block.
    """
    if block_length < 1 or replications < 100 or annualization_factor < 1:
        raise ValueError("block_length, replications, and annualization_factor must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    differential = _paired_differential(strategy_returns, benchmark_returns)
    if len(differential) < max(5, block_length):
        raise ValueError("Not enough aligned observations for the requested block length")
    observed_mean = float(differential.mean())
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(replications, dtype=float)
    values = differential.to_numpy(dtype=float)
    for replication in range(replications):
        sample = _moving_block_sample(values, block_length, rng)
        bootstrap_means[replication] = float(sample.mean())
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Centered bootstrap null: sample variation around the observed mean tests
    # whether a zero mean could plausibly generate the observed differential.
    null_means = bootstrap_means - observed_mean
    p_value = float((np.count_nonzero(null_means >= observed_mean) + 1) / (replications + 1))
    strategy_sharpe = _annualized_sharpe(pd.to_numeric(strategy_returns, errors="raise").dropna(), annualization_factor)
    benchmark_sharpe = _annualized_sharpe(pd.to_numeric(benchmark_returns, errors="raise").dropna(), annualization_factor)
    return BlockBootstrapComparison(
        observations=len(differential),
        block_length=block_length,
        replications=replications,
        annualization_factor=annualization_factor,
        mean_daily_excess_return=observed_mean,
        annualized_excess_return=float(observed_mean * annualization_factor),
        observed_annualized_sharpe_difference=strategy_sharpe - benchmark_sharpe,
        bootstrap_ci_low=float(ci_low),
        bootstrap_ci_high=float(ci_high),
        one_sided_p_value=p_value,
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_periodic_sharpes: Sequence[float],
) -> DeflatedSharpeResult:
    """Estimate the probability a periodic Sharpe exceeds the search-adjusted hurdle.

    ``trial_periodic_sharpes`` must contain every candidate configuration's
    validation Sharpe on the same return periodicity, including the selected
    candidate. It is a diagnostic against configuration mining, not a
    substitute for an outer held-out test.
    """
    values = pd.to_numeric(returns, errors="raise").dropna().to_numpy(dtype=float)
    trials = np.asarray(list(trial_periodic_sharpes), dtype=float)
    if len(values) < 3 or len(trials) < 2:
        raise ValueError("Deflated Sharpe requires at least three returns and two trial Sharpes")
    if not np.isfinite(values).all() or not np.isfinite(trials).all():
        raise ValueError("Returns and trial Sharpes must be finite")
    volatility = values.std(ddof=1)
    if volatility <= 0:
        raise ValueError("Returns need positive volatility for a Sharpe diagnostic")
    observed = float(values.mean() / volatility)
    expected_max = _expected_max_normal(float(trials.mean()), float(trials.std(ddof=1)), len(trials))
    centered = values - values.mean()
    skewness = float((centered**3).mean() / volatility**3)
    kurtosis = float((centered**4).mean() / volatility**4)
    denominator = np.sqrt(max(1e-12, 1.0 - skewness * observed + ((kurtosis - 1.0) / 4.0) * observed**2))
    z_score = (observed - expected_max) * np.sqrt(len(values) - 1) / denominator
    probability = float(NormalDist().cdf(z_score))
    return DeflatedSharpeResult(
        observations=len(values),
        trials=len(trials),
        observed_periodic_sharpe=observed,
        expected_max_trial_sharpe=expected_max,
        deflated_sharpe_probability=probability,
    )


def white_reality_check(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    block_length: int = 21,
    replications: int = 2_000,
    annualization_factor: int = 252,
    random_seed: int = 42,
) -> WhiteRealityCheckResult:
    """Test the best candidate after accounting for the complete search set.

    This uses White's centred bootstrap idea with common moving-block indices
    across all candidate-minus-benchmark daily differentials.  It controls the
    probability of selecting a seemingly positive candidate merely because a
    family of alternatives was searched.  Inputs must be out-of-sample returns
    from a single comparable validation or outer window; it is not a licence to
    combine runs with different data, costs, or calendar coverage.
    """
    if block_length < 1 or replications < 100 or annualization_factor < 1:
        raise ValueError("block_length, replications, and annualization_factor must be positive")
    if not isinstance(candidate_returns, pd.DataFrame) or candidate_returns.shape[1] < 2:
        raise ValueError("White Reality Check requires at least two candidate return series")
    if candidate_returns.columns.duplicated().any():
        raise ValueError("Candidate return series need unique column names")
    names = [str(name) for name in candidate_returns.columns]
    if any(not name.strip() for name in names):
        raise ValueError("Candidate return series need non-empty column names")
    if "__benchmark__" in names:
        raise ValueError("'__benchmark__' is reserved for the comparison series")
    candidates = candidate_returns.copy()
    candidates.columns = names
    aligned = pd.concat(
        [
            candidates.apply(pd.to_numeric, errors="raise"),
            pd.to_numeric(benchmark_returns, errors="raise").rename("__benchmark__"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < max(5, block_length):
        raise ValueError("Not enough common observations for the requested reality-check block length")
    values = aligned.loc[:, names].to_numpy(dtype=float)
    benchmark = aligned["__benchmark__"].to_numpy(dtype=float)
    differentials = values - benchmark[:, None]
    if not np.isfinite(differentials).all():
        raise ValueError("Candidate and benchmark returns must be finite")
    observed_means = differentials.mean(axis=0)
    best_index = int(np.argmax(observed_means))
    observed_best = float(observed_means[best_index])
    centered = differentials - observed_means[None, :]
    rng = np.random.default_rng(random_seed)
    bootstrap_best = np.empty(replications, dtype=float)
    for replication in range(replications):
        indices = _moving_block_indices(len(centered), block_length, rng)
        bootstrap_best[replication] = float(centered[indices].mean(axis=0).max())
    p_value = float((np.count_nonzero(bootstrap_best >= observed_best) + 1) / (replications + 1))
    return WhiteRealityCheckResult(
        observations=int(len(aligned)),
        candidates=int(len(names)),
        block_length=block_length,
        replications=replications,
        annualization_factor=annualization_factor,
        best_candidate=names[best_index],
        best_mean_daily_excess_return=observed_best,
        best_annualized_excess_return=float(observed_best * annualization_factor),
        reality_check_p_value=p_value,
    )


def _paired_differential(strategy: pd.Series, benchmark: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [pd.to_numeric(strategy, errors="raise"), pd.to_numeric(benchmark, errors="raise")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("Strategy and benchmark have no aligned finite return observations")
    differential = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if not np.isfinite(differential.to_numpy(dtype=float)).all():
        raise ValueError("Return differential must be finite")
    return differential


def _moving_block_sample(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    return values[_moving_block_indices(len(values), block_length, rng)]


def _moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks: list[np.ndarray] = []
    collected = 0
    while collected < length:
        start = int(rng.integers(0, length))
        blocks.append((start + np.arange(block_length)) % length)
        collected += block_length
    return np.concatenate(blocks)[:length]


def _annualized_sharpe(returns: pd.Series, annualization_factor: int) -> float:
    volatility = returns.std(ddof=1)
    return float(returns.mean() / volatility * np.sqrt(annualization_factor)) if volatility > 0 else 0.0


def _expected_max_normal(mean: float, std: float, trials: int) -> float:
    if std <= 0:
        return mean
    normal = NormalDist()
    # Euler-Mascheroni approximation used for the expected maximum of N
    # approximately normal trials (Bailey & López de Prado's DSR setup).
    euler_gamma = 0.5772156649015329
    first = normal.inv_cdf(1.0 - 1.0 / trials)
    second = normal.inv_cdf(1.0 - 1.0 / (trials * e))
    return mean + std * ((1.0 - euler_gamma) * first + euler_gamma * second)
