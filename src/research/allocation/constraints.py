"""Shared portfolio-weight validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_long_only_weights(weights: pd.Series, *, max_weight: float | None = None) -> pd.Series:
    """Validate a long-only target allocation with optional residual cash."""
    if weights.empty:
        raise ValueError("Portfolio target must contain at least one asset")
    validated = pd.to_numeric(weights, errors="raise").astype(float)
    if not np.isfinite(validated.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite")
    if (validated < -1e-12).any():
        raise ValueError("Long-only portfolio contains a negative weight")
    if validated.sum() > 1.0 + 1e-10:
        raise ValueError("Long-only portfolio weights must sum to at most 1.0")
    if max_weight is not None and (validated > max_weight + 1e-10).any():
        raise ValueError(f"Portfolio contains a weight above the {max_weight:.4f} cap")
    return validated.clip(lower=0.0)
