"""Build an auditable quality report for a candidate price/membership panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import build_panel_quality_report
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rebalance-frequency", default="ME")
    parser.add_argument("--return-outlier-threshold", type=float, default=0.20)
    parser.add_argument("--identity-crosswalk")
    parser.add_argument("--reference-prices")
    parser.add_argument("--reference-return-tolerance", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_paths = {
        "prices": _project_path(args.prices),
        "membership": _project_path(args.membership),
    }
    if args.identity_crosswalk:
        input_paths["identity_crosswalk"] = _project_path(args.identity_crosswalk)
    if args.reference_prices:
        input_paths["reference_prices"] = _project_path(args.reference_prices)
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")
    output = _project_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")

    report = build_panel_quality_report(
        pd.read_csv(input_paths["prices"]),
        pd.read_csv(input_paths["membership"]),
        rebalance_frequency=args.rebalance_frequency,
        return_outlier_threshold=args.return_outlier_threshold,
        identity_crosswalk=(
            pd.read_csv(input_paths["identity_crosswalk"]) if "identity_crosswalk" in input_paths else None
        ),
        reference_prices=(pd.read_csv(input_paths["reference_prices"]) if "reference_prices" in input_paths else None),
        reference_return_tolerance=args.reference_return_tolerance,
    )
    payload = {
        "kind": "candidate-panel-quality-report",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()
        },
        "rebalance_frequency": args.rebalance_frequency,
        "paper_eligible_by_itself": False,
        "warning": (
            "This is a diagnostic report. It does not validate total-return adjustments, source licensing, or "
            "point-in-time constituent methodology. Review every flagged item before a paper backtest."
        ),
        **report.as_jsonable(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote panel-quality report to {output}")


if __name__ == "__main__":
    main()
