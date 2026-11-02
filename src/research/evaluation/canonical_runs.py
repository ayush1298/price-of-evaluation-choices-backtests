"""Collect only manifest-backed canonical runs for paper tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def collect_canonical_runs(root: str | Path, *, include_exploratory: bool = False) -> pd.DataFrame:
    """Read completed canonical metrics without touching legacy artifacts.

    A directory contributes a row only when it contains both a
    ``kind=canonical-run-manifest`` file and a performance-metrics JSON file.
    This deliberately excludes graph-only smoke runs and all legacy outputs.
    Results are omitted unless their manifest explicitly declares
    ``paper_eligible=true``.  Reproducible-but-unclassified, public-data, and
    survivor-only development runs therefore cannot silently enter a paper
    table.  They can be retrieved for internal diagnostics with
    ``include_exploratory=True``.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    rows: list[dict[str, Any]] = []
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        manifest_path = directory / "run_manifest.json"
        metrics_path = directory / "performance_metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("kind") != "canonical-run-manifest":
            continue
        metrics = _read_json(metrics_path)
        row: dict[str, Any] = {
            "run_id": manifest.get("run_id"),
            "created_at_utc": manifest.get("created_at_utc"),
            "git_revision": manifest.get("git_revision"),
            "run_directory": str(directory),
            **{f"metric_{key}": value for key, value in metrics.items()},
        }
        extra = manifest.get("extra", {})
        if not isinstance(extra, dict):
            raise ValueError(f"Canonical manifest has non-object extra metadata: {manifest_path}")
        if extra.get("paper_eligible") is not True and not include_exploratory:
            continue
        row.update(_flatten_extra(extra))
        rows.append(row)
    return pd.DataFrame(rows)


def canonical_paper_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, traceable table sorted by net Sharpe if available."""
    if runs.empty:
        return runs.copy()
    preferred = [
        "run_id",
        "created_at_utc",
        "extra_strategy_graph_method",
        "extra_strategy_selector",
        "extra_strategy_allocator",
        "metric_annualized_return",
        "metric_annualized_volatility",
        "metric_sharpe",
        "metric_max_drawdown",
        "metric_cvar_95_daily",
        "metric_n_observations",
        "git_revision",
    ]
    columns = [column for column in preferred if column in runs.columns]
    table = runs.loc[:, columns].copy()
    return table.sort_values("metric_sharpe", ascending=False, kind="stable") if "metric_sharpe" in table else table


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _flatten_extra(extra: dict[str, Any], prefix: str = "extra") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in extra.items():
        output_key = f"{prefix}_{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_extra(value, output_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[output_key] = value
        else:
            # Lists (for example feature names) remain readable JSON instead
            # of being expanded into an unstable set of table columns.
            flattened[output_key] = json.dumps(value, sort_keys=True)
    return flattened
