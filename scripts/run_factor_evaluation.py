"""Run a manifest-backed Fama-French factor evaluation of a canonical NAV run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.evaluation import fit_factor_model
from src.research.reproducibility import (
    input_paths_with_existing_source_sidecars,
    sha256_file,
    write_run_manifest,
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        content = json.load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return content


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strategy-run", required=True, help="Canonical strategy run directory")
    parser.add_argument("--factors", required=True, help="Decimal daily Fama-French factor CSV")
    parser.add_argument("--config", default="configs/public_us_etf_2010_factor_evaluation.json")
    args = parser.parse_args()

    strategy_directory = _project_path(args.strategy_run)
    strategy_manifest_path = strategy_directory / "run_manifest.json"
    daily_nav_path = strategy_directory / "daily_nav.csv"
    if not strategy_manifest_path.exists() or not daily_nav_path.exists():
        raise FileNotFoundError("strategy-run must contain run_manifest.json and daily_nav.csv")
    strategy_manifest = _read_json(strategy_manifest_path)
    if strategy_manifest.get("kind") != "canonical-run-manifest":
        raise ValueError("strategy-run must be a canonical manifest-backed run")
    canonical = _read_json(_project_path(args.config))
    factor_columns = canonical.get("factor_columns")
    if not isinstance(factor_columns, list) or not all(isinstance(column, str) for column in factor_columns):
        raise ValueError("factor_columns must be a list of strings")
    risk_free_column = canonical.get("risk_free_column", "RF")
    if not isinstance(risk_free_column, str):
        raise ValueError("risk_free_column must be a string")
    factors_path = _project_path(args.factors)
    daily = pd.read_csv(daily_nav_path, parse_dates=["date"]).set_index("date")
    if "net_return" not in daily.columns:
        raise ValueError("Strategy daily NAV must contain net_return")
    result = fit_factor_model(
        daily["net_return"],
        pd.read_csv(factors_path, parse_dates=["date"]),
        factor_columns=factor_columns,
        risk_free_column=risk_free_column,
        annualization_factor=int(canonical.get("annualization_factor", 252)),
        hac_lags=int(canonical.get("hac_lags", 5)),
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars(
            [daily_nav_path, strategy_manifest_path, factors_path]
        ),
        extra={
            "analysis_kind": "fama_french_factor_evaluation",
            "strategy_run_id": strategy_manifest.get("run_id"),
            "strategy_git_revision": strategy_manifest.get("git_revision"),
            "strategy_manifest_sha256": sha256_file(strategy_manifest_path),
        },
    )
    (manifest_path.parent / "factor_regression.json").write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved factor evaluation to {manifest_path.parent.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
