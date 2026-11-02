"""Validate a candidate point-in-time market dataset before canonical runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import validate_point_in_time_market_data
from src.research.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--market", required=True, help="For example US, IN, EU, or JP")
    parser.add_argument("--currency", required=True, help="Local reporting currency, for example USD or INR")
    parser.add_argument("--rebalance-frequency", default="ME")
    parser.add_argument("--minimum-eligible-assets", type=int, default=30)
    parser.add_argument("--maximum-membership-staleness-days", type=int, default=45)
    parser.add_argument("--require-ohlcv", action="store_true")
    parser.add_argument("--output", help="Optional JSON validation report path")
    args = parser.parse_args()

    report = validate_point_in_time_market_data(
        pd.read_csv(args.prices),
        pd.read_csv(args.universe),
        market=args.market,
        currency=args.currency,
        rebalance_frequency=args.rebalance_frequency,
        minimum_eligible_assets=args.minimum_eligible_assets,
        maximum_membership_staleness_days=args.maximum_membership_staleness_days,
        require_ohlcv=args.require_ohlcv,
    )
    output = {
        "kind": "point-in-time-market-data-report",
        **report.as_jsonable(),
        "inputs": {
            "prices": {"path": str(args.prices), "sha256": sha256_file(args.prices)},
            "universe": {"path": str(args.universe), "sha256": sha256_file(args.universe)},
        },
    }
    content = json.dumps(output, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        try:
            display_path = destination.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = destination
        print(f"Validated {args.market} data and wrote {display_path}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
