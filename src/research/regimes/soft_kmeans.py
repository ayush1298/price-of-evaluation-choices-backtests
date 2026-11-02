"""Causal soft K-Means regime probabilities for stable model conditioning.

Unlike a Gaussian HMM, this encoder does not estimate a transition matrix.  At
each decision date it fits only the scaler and K-Means centres on preceding
market observations, then turns the current observation's distances to those
centres into a probability vector.  It is therefore a stable, transparent
regime-feature baseline for a downstream forecasting model rather than a
retrospective state label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ExpandingSoftKMeansRegime:
    """Emit causally fit, semantically ordered soft cluster probabilities."""

    def __init__(
        self,
        n_components: int = 3,
        min_train_size: int = 60,
        random_state: int = 42,
        temperature: float = 1.0,
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        if min_train_size < n_components:
            raise ValueError("min_train_size must be at least n_components")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.n_components = n_components
        self.min_train_size = min_train_size
        self.random_state = random_state
        self.temperature = temperature

    def predict_expanding(
        self,
        features: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        date_column: str = "date",
        ordering_feature: str | None = None,
    ) -> pd.DataFrame:
        """Return one soft state vector per date after causal training warm-up."""
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
            model = KMeans(
                n_clusters=self.n_components,
                n_init=20,
                random_state=self.random_state,
            ).fit(scaled_train)
            current = scaler.transform(values[index : index + 1])[0]
            squared_distances = np.square(model.cluster_centers_ - current).sum(axis=1)
            assigned_distances = np.square(
                scaled_train - model.cluster_centers_[model.labels_]
            ).sum(axis=1)
            scale = max(float(np.median(assigned_distances)), np.finfo(float).eps)
            probabilities_raw = _softmax(-squared_distances / (self.temperature * scale))
            state_means = scaler.inverse_transform(model.cluster_centers_)[:, ordering_index]
            semantic_order = np.argsort(state_means, kind="stable")
            probabilities = probabilities_raw[semantic_order]
            record: dict[str, object] = {
                date_column: data.at[index, date_column],
                "state": int(np.argmax(probabilities)),
                "training_end": data.at[index - 1, date_column],
                "ordering_feature": ordering_feature,
                "distance_scale": scale,
            }
            for state, probability in enumerate(probabilities):
                record[f"probability_state_{state}"] = float(probability)
            records.append(record)
        return pd.DataFrame(records)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


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
    if not np.isfinite(data.loc[:, feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Regime features must be finite")
    return data
