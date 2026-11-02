"""Transparent ensembles of canonical target-weight streams."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def equally_average_target_weights(target_streams: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average fully aligned long-only target streams across model members.

    This is deliberately an *ex-ante* equal ensemble: each source must supply
    exactly the same (signal date, asset) target pairs, so missing output is
    never silently turned into a zero allocation. The caller remains
    responsible for using only model members fixed before the evaluation date.
    """
    if len(target_streams) < 2:
        raise ValueError("At least two target streams are required for an ensemble")
    prepared = [_prepare_targets(targets) for targets in target_streams]
    reference = prepared[0].index
    if any(not targets.index.equals(reference) for targets in prepared[1:]):
        raise ValueError("Every ensemble member must have identical signal-date and symbol target pairs")
    weights = np.column_stack([targets["weight"].to_numpy(dtype=float) for targets in prepared]).mean(axis=1)
    output = prepared[0].reset_index().loc[:, ["signal_date", "symbol"]]
    output["weight"] = weights
    grouped_totals = output.groupby("signal_date", sort=False)["weight"].sum()
    if (grouped_totals > 1.0 + 1e-10).any():
        raise ValueError("Averaged long-only weights exceed a total allocation of one")
    return output


def _prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "symbol", "weight"}
    if missing := required.difference(targets.columns):
        raise ValueError(f"Target stream is missing columns: {sorted(missing)}")
    prepared = targets.loc[:, ["signal_date", "symbol", "weight"]].copy()
    prepared["signal_date"] = pd.to_datetime(prepared["signal_date"], errors="raise")
    prepared["symbol"] = prepared["symbol"].astype(str)
    prepared["weight"] = pd.to_numeric(prepared["weight"], errors="raise")
    if prepared.duplicated(["signal_date", "symbol"]).any():
        raise ValueError("Target stream has duplicate signal-date and symbol rows")
    if not np.isfinite(prepared["weight"].to_numpy(dtype=float)).all() or (prepared["weight"] < -1e-12).any():
        raise ValueError("Target stream weights must be finite and long-only")
    totals = prepared.groupby("signal_date", sort=False)["weight"].sum()
    if (totals > 1.0 + 1e-10).any():
        raise ValueError("Target stream has a total allocation above one")
    return prepared.set_index(["signal_date", "symbol"]).sort_index()
