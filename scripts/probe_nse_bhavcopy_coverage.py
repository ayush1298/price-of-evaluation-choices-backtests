"""Sample official NSE bhavcopies to audit historical symbol coverage.

This deliberately produces an *unadjusted coverage probe*, not a backtest
panel.  It makes the otherwise hidden delisted-symbol question measurable
before investing in a full archive download and corporate-action reconciliation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import business_days_on_or_before, fetch_nse_equity_bhavcopy
from src.research.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="CSV with NSE symbols to audit")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--start", required=True, help="First calendar month to sample")
    parser.add_argument("--end", required=True, help="Last calendar month to sample")
    parser.add_argument("--output", required=True, help="Long-format monthly coverage observations")
    parser.add_argument(
        "--max-prior-business-days",
        type=int,
        default=5,
        help="Retry this many prior business days when calendar month-end is closed",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument(
        "--include-isin",
        action="store_true",
        help="Retain exchange-reported ISIN for identity audit; does not adjust prices",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_prior_business_days < 0 or args.pause_seconds < 0:
        raise ValueError("max-prior-business-days and pause-seconds must be non-negative")
    symbols_path = Path(args.symbols)
    output_path = Path(args.output)
    if not symbols_path.exists():
        raise FileNotFoundError(symbols_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; review it and pass --overwrite if intended")
    symbols_frame = pd.read_csv(symbols_path)
    if args.symbol_column not in symbols_frame.columns:
        raise ValueError(f"Symbols file is missing column {args.symbol_column!r}")
    symbols = tuple(
        symbol
        for symbol in symbols_frame[args.symbol_column].astype(str).str.strip().str.upper().tolist()
        if symbol
    )
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Symbols must be unique, non-empty NSE identifiers")

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("end must not precede start")
    month_ends = pd.date_range(start=start, end=end, freq="ME")
    if month_ends.empty:
        raise ValueError("No calendar month ends fall within the requested interval")

    observations: list[pd.DataFrame] = []
    snapshots: list[dict[str, object]] = []
    for month_end in month_ends:
        fetches = []
        resolved = False
        for candidate in business_days_on_or_before(
            month_end, maximum_prior_business_days=args.max_prior_business_days
        ):
            if candidate < start:
                break
            result = fetch_nse_equity_bhavcopy(candidate, symbols=symbols, include_isin=args.include_isin)
            fetches.append(result)
            if result.succeeded:
                assert result.panel is not None
                observations.append(result.panel)
                found = sorted(result.panel["symbol"].unique().tolist())
                snapshots.append(
                    {
                        "requested_month_end": month_end.date().isoformat(),
                        "archive_date": candidate.date().isoformat(),
                        "url": result.url,
                        "raw_sha256": result.raw_sha256,
                        "found_symbols": found,
                        "missing_symbols": sorted(set(symbols).difference(found)),
                        "attempt_errors": [fetch.error for fetch in fetches if fetch.error],
                    }
                )
                resolved = True
                break
            if args.pause_seconds:
                time.sleep(args.pause_seconds)
        if not resolved:
            snapshots.append(
                {
                    "requested_month_end": month_end.date().isoformat(),
                    "archive_date": None,
                    "url": None,
                    "raw_sha256": None,
                    "found_symbols": [],
                    "missing_symbols": list(symbols),
                    "attempt_errors": [fetch.error for fetch in fetches if fetch.error],
                }
            )

    if not observations:
        raise RuntimeError("No requested month had a readable NSE bhavcopy archive")
    panel = pd.concat(observations, ignore_index=True).sort_values(["date", "symbol"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata = {
        "kind": "official-nse-equity-bhavcopy-monthly-coverage-probe",
        "provider": "NSE historical equity bhavcopy archive",
        "archive_base_url": "https://archives.nseindia.com/content/historical/EQUITIES",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_file": str(symbols_path),
        "symbols_file_sha256": sha256_file(symbols_path),
        "symbol_column": args.symbol_column,
        "requested_symbols": list(symbols),
        "requested_months": int(len(month_ends)),
        "max_prior_business_days": args.max_prior_business_days,
        "date_selection_policy": "month-end or earlier only; never advances a weekend/holiday signal date",
        "n_observations": int(len(panel)),
        "n_symbols_observed": int(panel["symbol"].nunique()),
        "price_adjustment_basis": "raw_unadjusted_exchange_prices",
        "identity_provenance": "exchange_reported_isin" if args.include_isin else "not_requested",
        "permitted_use": "historical-symbol-coverage audit only",
        "not_permitted": [
            "merge_with_adjusted_yfinance_prices",
            "paper_backtest",
            "corporate-action-adjusted-return claim",
        ],
        "snapshots": snapshots,
    }
    Path(f"{output_path}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved {len(panel):,} unadjusted monthly NSE observations for "
        f"{panel['symbol'].nunique()} symbols to {output_path}"
    )


if __name__ == "__main__":
    main()
