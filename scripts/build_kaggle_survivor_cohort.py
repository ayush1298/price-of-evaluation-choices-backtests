"""Build a deterministic, sector-balanced current-S&P survivor cohort.

This command is exclusively for the Kaggle current-company source.  The output
is suitable for an explicitly labelled exploratory fixed-stock panel, not a
historical S&P 500 constituent study or a paper headline result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import select_sector_balanced_current_survivor_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, help="Imported Kaggle OHLCV CSV")
    parser.add_argument("--companies", required=True, help="Kaggle sp500_companies.csv")
    parser.add_argument("--start", default="2010-01-04")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--per-sector",
        type=int,
        default=8,
        help="Fixed number of complete-history current survivors per sector (8 supports every current sector)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cohort manifest: {output}")
    cohort = select_sector_balanced_current_survivor_cohort(
        pd.read_csv(args.prices, usecols=["date", "symbol"]),
        pd.read_csv(args.companies),
        start=args.start,
        end=args.end,
        per_sector=args.per_sector,
        require_current_index_entry_by_start=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    print(f"Saved {len(cohort)} current-survivor exploratory cohort symbols to {output}")


if __name__ == "__main__":
    main()
