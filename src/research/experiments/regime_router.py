"""Soft, probability-weighted routing between portfolio signal builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.allocation.constraints import validate_long_only_weights


SignalBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]


@dataclass
class ProbabilisticRegimeRouter:
    """Blend allocations from regime-specific builders using causal HMM probabilities.

    Hard regime switches make short state intervals extremely brittle.  This
    router instead calls each configured signal builder at every rebalance and
    blends their allocations by the observed state posterior.  Consequently a
    60/40 uncertainty between two states results in a 60/40 blend, rather than
    a discontinuous jump caused by an arbitrary argmax state label.
    """

    probabilities: pd.DataFrame
    builders: Mapping[int, SignalBuilder]
    max_weight: float | None = None
    date_column: str = "date"
    training_end_column: str = "training_end"
    diagnostics: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.builders:
            raise ValueError("At least one regime-specific signal builder is required")
        required = {self.date_column, self.training_end_column}
        missing = required.difference(self.probabilities.columns)
        if missing:
            raise ValueError(f"Regime probabilities are missing columns: {sorted(missing)}")
        data = self.probabilities.copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column], errors="raise")
        data[self.training_end_column] = pd.to_datetime(data[self.training_end_column], errors="raise")
        if data[self.date_column].duplicated().any():
            raise ValueError("Regime probabilities must have one row per signal date")
        if not (data[self.training_end_column] < data[self.date_column]).all():
            raise ValueError("Every regime prediction must train strictly before its signal date")
        expected_columns = {f"probability_state_{state}" for state in self.builders}
        probability_columns = {column for column in data.columns if column.startswith("probability_state_")}
        if probability_columns != expected_columns:
            raise ValueError(
                "Regime probability columns must exactly match the supplied builders; "
                f"expected {sorted(expected_columns)}, got {sorted(probability_columns)}"
            )
        values = data.loc[:, sorted(probability_columns)].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any().any():
            raise ValueError("Regime probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Regime probabilities must sum to one at every signal date")
        self.probabilities = data.set_index(self.date_column).sort_index()

    def __call__(self, history: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
        date = pd.Timestamp(signal_date)
        if date not in self.probabilities.index:
            raise ValueError(f"No causal regime probabilities are available for {date.date()}")
        row = self.probabilities.loc[date]
        blended = pd.Series(dtype=float)
        state_probabilities: dict[int, float] = {}
        for state, builder in sorted(self.builders.items()):
            probability = float(row[f"probability_state_{state}"])
            candidate = builder(history, date)
            if not isinstance(candidate, pd.Series):
                raise TypeError(f"Builder for state {state} did not return a pandas Series")
            candidate = validate_long_only_weights(candidate, max_weight=self.max_weight)
            blended = blended.add(candidate.mul(probability), fill_value=0.0)
            state_probabilities[state] = probability
        blended = validate_long_only_weights(blended.sort_index(), max_weight=self.max_weight)
        self.diagnostics.append(
            {
                "signal_date": date,
                "training_end": row[self.training_end_column],
                "selected_state": int(max(state_probabilities, key=state_probabilities.get)),
                "selected_state_probability": max(state_probabilities.values()),
                "n_assets_after_blend": len(blended),
                **{f"probability_state_{state}": value for state, value in state_probabilities.items()},
            }
        )
        return blended

    def diagnostics_frame(self) -> pd.DataFrame:
        """Return one audit row per probability-weighted target allocation."""
        return pd.DataFrame(self.diagnostics)
