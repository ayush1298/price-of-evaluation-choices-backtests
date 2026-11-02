"""Causal, probabilistic Gaussian-HMM regime inference.

This is a stronger regime baseline than the legacy all-sample K-Means labels.
At decision date ``t``, both the scaler and HMM parameters use observations
strictly before ``t``.  The state posterior at ``t`` uses the current feature
vector and the preceding sequence only; it never smooths with future data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class ExpandingGaussianHMMRegime:
    """Refit a Gaussian HMM on an expanding history and emit state probabilities.

    HMM state identifiers are otherwise arbitrary after every refit.  To make
    an experiment auditable, output states are re-labelled by ascending mean
    of a declared ``ordering_feature`` (normally realised volatility). Thus
    ``state_0`` is the lowest-volatility inferred state within each historical
    fit, rather than a cluster ID with accidental semantics.
    """

    def __init__(
        self,
        n_components: int = 3,
        min_train_size: int = 60,
        random_state: int = 42,
        n_iter: int = 200,
        covariance_type: str = "diag",
        startprob_prior: float = 1.1,
        transmat_prior: float = 1.1,
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        if min_train_size < n_components * 2:
            raise ValueError("min_train_size must be at least twice n_components")
        if n_iter < 1:
            raise ValueError("n_iter must be positive")
        if covariance_type not in {"diag", "full", "spherical", "tied"}:
            raise ValueError(f"Unsupported HMM covariance_type: {covariance_type}")
        if startprob_prior < 1.0 or transmat_prior < 1.0:
            raise ValueError("HMM start and transition priors must be at least 1.0")
        self.n_components = n_components
        self.min_train_size = min_train_size
        self.random_state = random_state
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior

    def predict_expanding(
        self,
        features: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        date_column: str = "date",
        ordering_feature: str | None = None,
    ) -> pd.DataFrame:
        """Produce causally valid state probabilities for eligible dates.

        The feature vector observed after a signal-date close may be used to
        classify that date's regime.  A portfolio action based on the state
        must still execute on a later tradable session in the backtest engine.
        """
        if not feature_columns:
            raise ValueError("At least one regime feature is required")
        if ordering_feature is None:
            ordering_feature = feature_columns[0]
        if ordering_feature not in feature_columns:
            raise ValueError("ordering_feature must be included in feature_columns")
        data = _validate_features(features, feature_columns, date_column)
        values = data.loc[:, feature_columns].to_numpy(dtype=float)
        ordering_index = list(feature_columns).index(ordering_feature)
        records: list[dict[str, object]] = []

        for index in range(self.min_train_size, len(data)):
            train_values = values[:index]
            scaler = StandardScaler().fit(train_values)
            scaled_train = scaler.transform(train_values)
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                # hmmlearn subtracts one from these Dirichlet priors in its
                # M-step.  A value strictly above one prevents an otherwise
                # unvisited state from acquiring an all-zero transition row
                # in short expanding financial samples.
                startprob_prior=self.startprob_prior,
                transmat_prior=self.transmat_prior,
            ).fit(scaled_train)
            # The classified observation is last, so its posterior is a
            # filtering posterior: no later observation exists to smooth from.
            scaled_sequence = np.vstack([scaled_train, scaler.transform(values[index : index + 1])])
            raw_probabilities = model.predict_proba(scaled_sequence)[-1]
            state_means = scaler.inverse_transform(model.means_)[:, ordering_index]
            semantic_order = np.argsort(state_means, kind="stable")
            probabilities = raw_probabilities[semantic_order]
            record: dict[str, object] = {
                date_column: data.at[index, date_column],
                "state": int(np.argmax(probabilities)),
                "training_end": data.at[index - 1, date_column],
                "ordering_feature": ordering_feature,
            }
            for state, probability in enumerate(probabilities):
                record[f"probability_state_{state}"] = float(probability)
            records.append(record)
        return pd.DataFrame(records)


def _validate_features(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    date_column: str,
) -> pd.DataFrame:
    required = [date_column, *feature_columns]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Missing regime feature columns: {missing}")
    data = features.loc[:, required].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="raise")
    data = data.sort_values(date_column).reset_index(drop=True)
    if data[date_column].duplicated().any():
        raise ValueError("Regime features must have one row per decision date")
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    values = data.loc[:, feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Regime features must be finite")
    return data
