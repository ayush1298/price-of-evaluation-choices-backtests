"""Build auditable rebalance-date membership for a pre-declared ETF universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data.fixed_universe import build_fixed_universe_membership
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, help="Canonical long-format public price CSV")
    parser.add_argument("--symbols", required=True, help="CSV manifest with a symbol column")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--output", required=True, help="Membership CSV to create")
    parser.add_argument("--rebalance-frequency", default="ME")
    parser.add_argument(
        "--allow-missing-rebalance-prices",
        action="store_true",
        help="Exploratory mode only: mark missing assets ineligible instead of failing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prices_path = _project_path(args.prices)
    symbols_path = _project_path(args.symbols)
    output_path = _project_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; review it and pass --overwrite if intended")
    manifest = pd.read_csv(symbols_path)
    if args.symbol_column not in manifest.columns:
        raise ValueError(f"Symbol manifest is missing {args.symbol_column!r}")
    membership = build_fixed_universe_membership(
        pd.read_csv(prices_path),
        manifest[args.symbol_column],
        rebalance_frequency=args.rebalance_frequency,
        require_complete_prices=not args.allow_missing_rebalance_prices,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_path, index=False)
    source_path = output_path.with_suffix(output_path.suffix + ".source.json")
    source_path.write_text(
        json.dumps(
            {
                "kind": "fixed-declared-universe-membership",
                "universe_definition": "assets declared in the supplied manifest before data extraction",
                "prices_path": str(prices_path),
                "prices_sha256": sha256_file(prices_path),
                "symbols_path": str(symbols_path),
                "symbols_sha256": sha256_file(symbols_path),
                "rebalance_frequency": args.rebalance_frequency,
                "require_complete_rebalance_prices": not args.allow_missing_rebalance_prices,
                "membership_rows": len(membership),
                "membership_dates": int(membership["date"].nunique()),
                "declared_assets": int(manifest[args.symbol_column].nunique()),
                "warning": (
                    "This static membership construction is valid only for a pre-declared fixed universe "
                    "such as ETFs; it is not historical stock-index membership."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(membership):,} membership rows across {membership['date'].nunique()} snapshots to {output_path}")


if __name__ == "__main__":
    main()
