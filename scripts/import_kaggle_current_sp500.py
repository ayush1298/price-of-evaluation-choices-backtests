"""Import a manually downloaded current-S&P Kaggle price file with provenance.

The source is useful for long-horizon model development, but it contains the
current S&P 500 roster, not historical membership including removals.  The
resulting sidecar therefore marks it as an exploratory survivor-only panel.
This command intentionally does not fabricate a point-in-time membership file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import (
    drop_incomplete_ohlcv_dates,
    drop_symbols_with_materially_invalid_ohlcv,
    normalise_kaggle_current_sp500_stocks,
    validate_price_panel,
)
from src.research.reproducibility import sha256_file


DEFAULT_SOURCE_URL = "https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", required=True, help="Manually downloaded sp500_stocks.csv")
    parser.add_argument("--output", required=True, help="Output long-format OHLCV CSV")
    parser.add_argument("--dataset-version", required=True, type=int, help="Kaggle dataset version shown at download")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--close-adjustment-convention",
        required=True,
        help="Exact provider convention, e.g. 'unadjusted_close_as_downloaded'; never guess",
    )
    parser.add_argument(
        "--drop-incomplete-dates",
        action="store_true",
        help="Drop every date containing a missing OHLCV field; records dates rather than imputing values",
    )
    parser.add_argument(
        "--drop-symbols-with-invalid-ohlcv",
        action="store_true",
        help="Remove every symbol with a materially inconsistent OHLCV bar; records symbols without modifying prices",
    )
    args = parser.parse_args()

    source = Path(args.stocks)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    raw = pd.read_csv(source)
    panel = normalise_kaggle_current_sp500_stocks(raw)
    dropped_dates: list[str] = []
    if args.drop_incomplete_dates:
        panel, incomplete_dates = drop_incomplete_ohlcv_dates(panel)
        dropped_dates = [date.isoformat() for date in incomplete_dates]
    dropped_symbols: list[str] = []
    if args.drop_symbols_with_invalid_ohlcv:
        panel, invalid_symbols = drop_symbols_with_materially_invalid_ohlcv(panel)
        dropped_symbols = list(invalid_symbols)
    validated, report = validate_price_panel(panel, require_ohlcv=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False)
    metadata = {
        "kind": "public-kaggle-current-sp500-survivor-panel",
        "source": "Kaggle darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily",
        "source_url": args.source_url,
        "dataset_version": args.dataset_version,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_stocks_path": str(source),
        "raw_stocks_sha256": sha256_file(source),
        "close_adjustment_convention": args.close_adjustment_convention,
        "incomplete_ohlcv_date_policy": "drop_entire_date" if args.drop_incomplete_dates else "reject",
        "dropped_incomplete_ohlcv_dates": dropped_dates,
        "invalid_ohlcv_symbol_policy": "drop_entire_symbol" if args.drop_symbols_with_invalid_ohlcv else "reject",
        "dropped_symbols_with_materially_invalid_ohlcv": dropped_symbols,
        "universe_scope": "survivor_only_current_constituents_not_historical_sp500_membership",
        "eligible_for_canonical_historical_index_claim": False,
        "rows": int(len(validated)),
        "symbols": int(validated["symbol"].nunique()),
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
    }
    Path(f"{output}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Imported exploratory current-S&P Kaggle panel to {output}")


if __name__ == "__main__":
    main()
