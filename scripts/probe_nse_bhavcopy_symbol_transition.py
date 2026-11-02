"""Audit exact NSE trading-code transitions in a short daily window.

This is an identity-evidence tool, not a price-panel downloader. It records
which requested symbols appeared in each readable raw bhavcopy plus the source
hash/URL, allowing a crosswalk review to identify the observed final old-code
and first new-code trading days without inferring continuity from monthly data.
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

from src.research.data import fetch_nse_equity_bhavcopy
from src.research.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="CSV with the old/new NSE symbols")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--start", required=True, help="Inclusive calendar date")
    parser.add_argument("--end", required=True, help="Inclusive calendar date")
    parser.add_argument("--output", required=True, help="New daily transition-observation CSV")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    if args.pause_seconds < 0:
        raise ValueError("pause-seconds must be non-negative")
    source_path = Path(args.symbols)
    output_path = Path(args.output)
    if output_path.exists() or Path(f"{output_path}.source.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output or sidecar: {output_path}")
    source = pd.read_csv(source_path)
    if args.symbol_column not in source.columns:
        raise ValueError(f"{source_path} is missing {args.symbol_column!r}")
    symbols = tuple(
        dict.fromkeys(
            symbol
            for value in source[args.symbol_column]
            if (symbol := str(value).strip().upper()) and symbol.lower() != "nan"
        )
    )
    if not symbols:
        raise ValueError("No non-empty symbols supplied")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("Require valid dates with end >= start")

    observations: list[pd.DataFrame] = []
    snapshots: list[dict[str, object]] = []
    candidates = pd.date_range(start, end, freq="B")
    for index, date in enumerate(candidates):
        if index and args.pause_seconds:
            time.sleep(args.pause_seconds)
        result = fetch_nse_equity_bhavcopy(date, symbols=symbols)
        if result.succeeded:
            assert result.panel is not None
            observations.append(result.panel)
            found = sorted(result.panel["symbol"].unique().tolist())
            snapshots.append(
                {
                    "date": date.date().isoformat(),
                    "url": result.url,
                    "raw_sha256": result.raw_sha256,
                    "found_symbols": found,
                    "missing_symbols": sorted(set(symbols).difference(found)),
                    "error": None,
                }
            )
        else:
            snapshots.append(
                {
                    "date": date.date().isoformat(),
                    "url": result.url,
                    "raw_sha256": result.raw_sha256,
                    "found_symbols": [],
                    "missing_symbols": list(symbols),
                    "error": result.error,
                }
            )
    panel = (
        pd.concat(observations, ignore_index=True).sort_values(["date", "symbol"])
        if observations
        else pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    metadata = {
        "kind": "official-nse-equity-bhavcopy-daily-symbol-transition-probe",
        "provider": "NSE historical equity bhavcopy archive",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols_file": str(source_path),
        "symbols_file_sha256": sha256_file(source_path),
        "symbol_column": args.symbol_column,
        "requested_symbols": list(symbols),
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
        "business_day_candidates": int(len(candidates)),
        "readable_archives": int(sum(snapshot["error"] is None for snapshot in snapshots)),
        "observations": int(len(panel)),
        "price_adjustment_basis": "raw_unadjusted_exchange_prices",
        "permitted_use": "exact_ticker_transition_evidence_only",
        "not_permitted": ["corporate_action_adjusted_return_claim", "paper_backtest"],
        "snapshots": snapshots,
    }
    Path(f"{output_path}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(panel)} raw daily transition observations to {output_path}")


if __name__ == "__main__":
    main()
