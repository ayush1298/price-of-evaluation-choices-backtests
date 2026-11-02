"""Leakage-safe expanding-window K-Means regime baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegimePrediction:
    date: pd.Timestamp
    state: int
    training_end: pd.Timestamp


class ExpandingKMeansRegime:
    """Fit K-Means using only information known at each decision date.

    Cluster identifiers are deliberately left as ``state_0``/``state_1`` etc.
    Naming a state ``crisis`` or ``bull`` requires a separately specified
    training-period rule; historical anchor years must not determine labels.
    """

    def __init__(self, n_clusters: int = 3, min_train_size: int = 36, random_state: int = 42):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if min_train_size < n_clusters:
            raise ValueError("min_train_size must be at least n_clusters")
        self.n_clusters = n_clusters
        self.min_train_size = min_train_size
        self.random_state = random_state

    def predict_expanding(
        self,
        features: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Return one state prediction per eligible decision date.

        The observation at date ``t`` may be used to classify the regime known
        after the close of ``t``. The scaler and centroids use observations
        strictly before ``t``.
        """
        required = [date_column, *feature_columns]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise ValueError(f"Missing regime feature columns: {missing}")

        data = features.loc[:, required].copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="raise")
        data = data.sort_values(date_column).reset_index(drop=True)
        if data[date_column].duplicated().any():
            raise ValueError("Regime features must have one row per decision date")
        if data.loc[:, feature_columns].isna().any().any():
            raise ValueError("Regime features contain missing values")

        predictions: list[RegimePrediction] = []
        values = data.loc[:, feature_columns].to_numpy(dtype=float)
        for index in range(self.min_train_size, len(data)):
            train_values = values[:index]
            scaler = StandardScaler().fit(train_values)
            model = KMeans(
                n_clusters=self.n_clusters,
                n_init=20,
                random_state=self.random_state,
            ).fit(scaler.transform(train_values))
            state = int(model.predict(scaler.transform(values[index : index + 1]))[0])
            predictions.append(
                RegimePrediction(
                    date=data.at[index, date_column],
                    state=state,
                    training_end=data.at[index - 1, date_column],
                )
            )

        return pd.DataFrame(
            {
                date_column: [prediction.date for prediction in predictions],
                "state": [prediction.state for prediction in predictions],
                "training_end": [prediction.training_end for prediction in predictions],
            }
        )
