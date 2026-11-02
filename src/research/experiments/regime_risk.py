"""Causal soft-regime conditional covariance allocation experiments.

The model deliberately leaves the forecast stage out of the decision.  It uses
only past asset returns and causally inferred market-state probabilities to
estimate the covariance relevant to the *current* state, then uses the shared
long-only Hierarchical Risk Parity allocator.  This is a focused response to
the observed public-panel failure mode: unstable expected-return forecasts
were weaker than transparent risk-controlled baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from src.research.allocation import hierarchical_risk_parity_weights_from_covariance
from src.research.data import eligible_universe, validate_membership_history


@dataclass(frozen=True)
class RegimeConditionalRiskConfig:
    """Predeclared conditional-covariance and allocation choices."""

    lookback_sessions: int = 756
    minimum_assets: int = 25
    minimum_effective_observations: float = 21.0
    covariance_shrinkage: float = 0.25
    max_weight: float = 0.1
    uncertainty_to_unconditional: bool = False
    minimum_regime_confidence: float = 0.0
    require_causal_training_end: bool = False
    group_aware_within_group_only: bool = False

    def __post_init__(self) -> None:
        if self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be at least two")
        if self.minimum_assets < 2:
            raise ValueError("minimum_assets must be at least two")
        if self.minimum_effective_observations < 2:
            raise ValueError("minimum_effective_observations must be at least two")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if not 0 < self.max_weight <= 1 or self.minimum_assets * self.max_weight < 1 - 1e-12:
            raise ValueError("minimum_assets and max_weight cannot form a fully invested portfolio")
        if not isinstance(self.uncertainty_to_unconditional, bool):
            raise ValueError("uncertainty_to_unconditional must be a boolean")
        if not 0.0 <= self.minimum_regime_confidence < 1.0:
            raise ValueError("minimum_regime_confidence must be in [0, 1)")
        if not isinstance(self.require_causal_training_end, bool):
            raise ValueError("require_causal_training_end must be a boolean")
        if not isinstance(self.group_aware_within_group_only, bool):
            raise ValueError("group_aware_within_group_only must be a boolean")


@dataclass
class SoftRegimeConditionalHRPBuilder:
    """Make a point-in-time conditional-HRP target at each signal date.

    ``regime_probabilities`` must contain one causal probability vector at
    each available signal date.  A daily return is associated only with the
    latest probability vector known at its close; it never receives a later
    month-end label.  The current probability vector mixes historical
    state-specific covariance estimates at the decision date.
    """

    config: RegimeConditionalRiskConfig
    universe_history: pd.DataFrame
    regime_probabilities: pd.DataFrame
    asset_groups: Mapping[str, str] | None = None
    diagnostics: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.universe_history = validate_membership_history(self.universe_history)
        probabilities = self.regime_probabilities.copy()
        if "date" not in probabilities.columns:
            raise ValueError("Regime probabilities require a date column")
        columns = tuple(column for column in probabilities.columns if column.startswith("probability_state_"))
        if len(columns) < 2:
            raise ValueError("Regime probabilities require at least two state columns")
        keep_columns = ["date", *columns]
        if "training_end" in probabilities.columns:
            keep_columns.append("training_end")
        elif self.config.require_causal_training_end:
            raise ValueError("Regime probabilities require training_end for this configuration")
        probabilities = probabilities.loc[:, keep_columns].copy()
        probabilities["date"] = pd.to_datetime(probabilities["date"], errors="raise")
        if probabilities["date"].duplicated().any():
            raise ValueError("Regime probabilities require one row per date")
        if "training_end" in probabilities.columns:
            probabilities["training_end"] = pd.to_datetime(probabilities["training_end"], errors="raise")
            if not (probabilities["training_end"] < probabilities["date"]).all():
                raise ValueError("Every regime probability must train strictly before its decision date")
        probabilities.loc[:, columns] = probabilities.loc[:, columns].apply(pd.to_numeric, errors="raise")
        values = probabilities.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Regime probabilities must be finite, non-negative, and sum to one")
        self._probability_columns = columns
        self.regime_probabilities = probabilities.sort_values("date").reset_index(drop=True)
        groups = self.asset_groups or {}
        self.asset_groups = {
            str(symbol).strip(): str(group).strip()
            for symbol, group in groups.items()
            if str(symbol).strip() and str(group).strip()
        }
        if self.config.group_aware_within_group_only and not self.asset_groups:
            raise ValueError("Group-aware conditional risk requires a non-empty asset_groups mapping")

    def __call__(self, history: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
        returns = self._historical_returns(history, signal_date)
        current_probability = self._current_probability(signal_date)
        state_probabilities = self._return_state_probabilities(returns.index, signal_date)
        groups = self._asset_groups_for_returns(returns.columns)
        (
            covariance,
            effective,
            fallback_states,
            regime_entropy,
            regime_confidence,
            group_count,
            psd_adjustment,
        ) = _regime_mixture_covariance(
            returns,
            state_probabilities,
            current_probability,
            minimum_effective_observations=self.config.minimum_effective_observations,
            shrinkage=self.config.covariance_shrinkage,
            uncertainty_to_unconditional=self.config.uncertainty_to_unconditional,
            minimum_regime_confidence=self.config.minimum_regime_confidence,
            asset_groups=groups,
        )
        weights = hierarchical_risk_parity_weights_from_covariance(
            covariance, max_weight=self.config.max_weight
        )
        row: dict[str, object] = {
            "signal_date": pd.Timestamp(signal_date),
            "lookback_sessions": self.config.lookback_sessions,
            "n_assets": int(returns.shape[1]),
            "n_return_observations": int(len(returns)),
            "covariance_shrinkage": self.config.covariance_shrinkage,
            "minimum_effective_observations": self.config.minimum_effective_observations,
            "fallback_state_count": int(fallback_states),
            "effective_observations_min": float(effective.min()),
            "effective_observations_max": float(effective.max()),
            "regime_entropy": float(regime_entropy),
            "regime_confidence": float(regime_confidence),
            "uncertainty_to_unconditional": self.config.uncertainty_to_unconditional,
            "group_aware_within_group_only": self.config.group_aware_within_group_only,
            "conditional_group_count": int(group_count),
            "group_covariance_psd_adjustment_frobenius": float(psd_adjustment),
        }
        if "training_end" in self.regime_probabilities.columns:
            row["regime_training_end"] = pd.Timestamp(
                self.regime_probabilities.loc[
                    self.regime_probabilities["date"] == pd.Timestamp(signal_date), "training_end"
                ].iloc[0]
            )
        row.update({f"current_{column}": float(value) for column, value in current_probability.items()})
        self.diagnostics.append(row)
        return weights

    def _asset_groups_for_returns(self, columns: pd.Index) -> dict[str, str] | None:
        if not self.config.group_aware_within_group_only:
            return None
        missing = sorted(set(columns.astype(str)).difference(self.asset_groups))
        if missing:
            raise ValueError(f"Group-aware conditional risk has no group mapping for assets: {missing}")
        return {str(symbol): self.asset_groups[str(symbol)] for symbol in columns}

    def diagnostics_frame(self) -> pd.DataFrame:
        """Return the complete decision-time covariance diagnostics."""
        return pd.DataFrame(self.diagnostics)

    def _historical_returns(self, history: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
        required = {"date", "symbol", "close"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"Price history is missing columns: {sorted(missing)}")
        prices = history.loc[:, ["date", "symbol", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="raise")
        prices["symbol"] = prices["symbol"].astype(str)
        prices["close"] = pd.to_numeric(prices["close"], errors="raise")
        eligible = eligible_universe(self.universe_history, signal_date)
        matrix = prices.pivot(index="date", columns="symbol", values="close").sort_index()
        candidates = sorted(set(eligible).intersection(matrix.columns.astype(str)))
        if len(candidates) < self.config.minimum_assets:
            raise ValueError(f"Only {len(candidates)} eligible assets have observed prices; need {self.config.minimum_assets}")
        window = matrix.reindex(columns=candidates).tail(self.config.lookback_sessions + 1)
        complete = window.columns[window.notna().all()].tolist()
        if len(complete) < self.config.minimum_assets:
            raise ValueError(
                "Insufficient continuous point-in-time price history for conditional covariance: "
                f"found {len(complete)}, need {self.config.minimum_assets}"
            )
        returns = window.loc[:, complete].pct_change(fill_method=None).iloc[1:]
        if len(returns) < self.config.lookback_sessions:
            raise ValueError(f"Only {len(returns)} returns are available; need {self.config.lookback_sessions}")
        if not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise ValueError("Historical return window contains non-finite values")
        return returns

    def _current_probability(self, signal_date: pd.Timestamp) -> pd.Series:
        matches = self.regime_probabilities.loc[self.regime_probabilities["date"] == pd.Timestamp(signal_date)]
        if len(matches) != 1:
            raise ValueError(f"No exact causal regime probability for signal date {pd.Timestamp(signal_date).date()}")
        return matches.iloc[0].loc[list(self._probability_columns)].astype(float)

    def _return_state_probabilities(
        self,
        return_dates: pd.DatetimeIndex,
        signal_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # A return ending on a non-signal date uses the most recent completed
        # signal state.  At a month-end close, the state inferred from that
        # close is usable because allocation executes only on a later session.
        historical = self.regime_probabilities.loc[
            self.regime_probabilities["date"] <= pd.Timestamp(signal_date), ["date", *self._probability_columns]
        ].copy()
        left = pd.DataFrame({"date": pd.DatetimeIndex(return_dates)})
        aligned = pd.merge_asof(
            left.sort_values("date"),
            historical.sort_values("date"),
            on="date",
            direction="backward",
        ).set_index("date")
        columns = [column for column in aligned.columns if column.startswith("probability_state_")]
        return aligned.loc[:, columns].dropna(how="any")


def _regime_mixture_covariance(
    returns: pd.DataFrame,
    state_probabilities: pd.DataFrame,
    current_probability: pd.Series,
    *,
    minimum_effective_observations: float,
    shrinkage: float,
    uncertainty_to_unconditional: bool,
    minimum_regime_confidence: float,
    asset_groups: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, int, float, float, int, float]:
    """Estimate a current-state covariance mixture using only aligned history."""
    aligned = returns.join(state_probabilities, how="inner")
    state_columns = list(state_probabilities.columns)
    if len(aligned) < 2:
        raise ValueError("Conditional covariance needs at least two aligned historical returns")
    asset_columns = list(returns.columns)
    values = aligned.loc[:, asset_columns].to_numpy(dtype=float)
    unconditional = pd.DataFrame(np.cov(values, rowvar=False, ddof=1), index=asset_columns, columns=asset_columns)
    if unconditional.shape != (len(asset_columns), len(asset_columns)):
        raise ValueError("Unconditional covariance has an invalid shape")
    unconditional_values = unconditional.to_numpy(dtype=float)
    mixtures = np.zeros_like(unconditional_values)
    effective_records: dict[str, float] = {}
    fallback_states = 0
    for state in state_columns:
        raw = aligned[state].to_numpy(dtype=float)
        effective = float(raw.sum() ** 2 / np.square(raw).sum()) if np.square(raw).sum() > 0 else 0.0
        effective_records[state] = effective
        if effective < minimum_effective_observations:
            covariance = unconditional_values
            fallback_states += 1
        else:
            covariance = _weighted_covariance(values, raw)
            covariance = (1.0 - shrinkage) * covariance + shrinkage * unconditional_values
        mixtures += float(current_probability[state]) * covariance
    current_values = current_probability.reindex(state_columns).to_numpy(dtype=float)
    positive = current_values[current_values > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    maximum_entropy = float(np.log(len(state_columns)))
    raw_confidence = 1.0 - entropy / maximum_entropy if maximum_entropy > 0 else 1.0
    if uncertainty_to_unconditional:
        # The model may not pretend that an almost-uniform state posterior is
        # an informative regime signal.  Below the declared confidence floor,
        # use the unconditional covariance; above it, phase in the conditional
        # covariance continuously rather than hard-switching portfolios.
        confidence = max(
            0.0,
            (raw_confidence - minimum_regime_confidence) / (1.0 - minimum_regime_confidence),
        )
        mixtures = confidence * mixtures + (1.0 - confidence) * unconditional_values
    else:
        confidence = 1.0
    group_count = 0
    psd_adjustment = 0.0
    if asset_groups is not None:
        group_labels = [asset_groups[str(asset)] for asset in asset_columns]
        group_count = len(set(group_labels))
        same_group = np.equal.outer(group_labels, group_labels)
        group_aware = np.where(same_group, mixtures, unconditional_values)
        projected = _project_positive_semidefinite(group_aware)
        psd_adjustment = float(np.linalg.norm(projected - group_aware, ord="fro"))
        mixtures = projected
    mixture = pd.DataFrame(mixtures, index=asset_columns, columns=asset_columns)
    return mixture, pd.Series(effective_records), fallback_states, entropy, confidence, group_count, psd_adjustment


def _weighted_covariance(values: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
    """Return a positive-semidefinite weighted sample covariance matrix."""
    weights = raw_weights / raw_weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    denominator = 1.0 - float(np.square(weights).sum())
    if denominator <= np.finfo(float).eps:
        raise ValueError("Insufficient effective observations for weighted covariance")
    return (centered * weights[:, None]).T @ centered / denominator


def _project_positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    """Symmetrise and clip only negative eigenvalues introduced by block mixing."""
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(float(np.diag(symmetric).mean()) * 1e-12, np.finfo(float).eps)
    clipped = np.clip(eigenvalues, floor, None)
    return (eigenvectors * clipped) @ eigenvectors.T
