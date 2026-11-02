"""Retrieve a bounded, auditable NSE corporate-action review feed.

The output is an event ledger, not a price adjustment file.  Analysts must
review the event meaning and explicitly create a supported price-scale action
record before `adjust_price_scale_actions.py` can modify raw bhavcopy prices.
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

from src.research.data import fetch_nse_corporate_actions, reconcile_nse_corporate_action_records


def _month_windows(start: pd.Timestamp, end: pd.Timestamp) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    starts = pd.date_range(start.to_period("M").start_time, end.to_period("M").start_time, freq="MS")
    return tuple((max(month_start, start), min(month_start + pd.offsets.MonthEnd(0), end)) for month_start in starts)


def _read_symbols(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    source = pd.read_csv(path)
    if "symbol" not in source.columns:
        raise ValueError(f"{path} must contain a symbol column")
    return tuple(source["symbol"].dropna().astype(str).str.strip().str.upper().loc[lambda x: x.ne("")].unique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="New normalized action-ledger CSV path")
    parser.add_argument("--symbols-csv", help="Optional CSV with a symbol column to restrict the review queue")
    parser.add_argument("--timeout-seconds", default=30.0, type=float)
    parser.add_argument(
        "--pause-seconds",
        default=0.2,
        type=float,
        help="Delay between monthly public-API requests (default: 0.2)",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("Require valid inclusive start/end dates with end >= start")
    if args.pause_seconds < 0:
        raise ValueError("pause-seconds must be non-negative")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    symbols = _read_symbols(Path(args.symbols_csv)) if args.symbols_csv else None

    action_frames: list[pd.DataFrame] = []
    fetches: list[dict[str, object]] = []
    errors: list[str] = []
    for index, (window_start, window_end) in enumerate(_month_windows(start, end)):
        if index:
            time.sleep(args.pause_seconds)
        result = fetch_nse_corporate_actions(
            window_start,
            window_end,
            symbols=symbols,
            timeout_seconds=args.timeout_seconds,
        )
        fetches.append(
            {
                "start": window_start.date().isoformat(),
                "end": window_end.date().isoformat(),
                "url": result.url,
                "raw_sha256": result.raw_sha256,
                "rows": None if result.events is None else int(len(result.events)),
                "error": result.error,
            }
        )
        if not result.succeeded:
            errors.append(f"{window_start.date()}–{window_end.date()}: {result.error}")
            continue
        action_frames.append(result.events)
    if errors:
        raise RuntimeError("NSE corporate-action retrieval failed; no partial ledger written:\n" + "\n".join(errors))

    nonempty_frames = [frame for frame in action_frames if not frame.empty]
    # Preserve the normalised empty schema while avoiding Pandas' ambiguous
    # all-NA concatenation behaviour for months with no action records.
    ledger = pd.concat(nonempty_frames, ignore_index=True) if nonempty_frames else action_frames[0].copy()
    ledger, reconciliation = reconcile_nse_corporate_action_records(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    metadata = {
        "kind": "official-nse-equity-corporate-actions-review-ledger",
        "source": "NSE corporate actions API",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
        "symbol_filter": None if symbols is None else list(symbols),
        "monthly_fetches": fetches,
        "rows": reconciliation.unique_records,
        "input_rows": reconciliation.input_records,
        "duplicate_records_collapsed": reconciliation.duplicate_records_collapsed,
        "duplicate_policy": "collapse_only_identical_normalised_source_records",
        "price_adjustment_status": "unreviewed_events_not_price_adjustments",
        "paper_eligibility": "requires_action_and_identity_review_before_use_in_adjusted_panel",
    }
    Path(f"{output}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(ledger)} review-only NSE corporate-action records to {output}")


if __name__ == "__main__":
    main()
