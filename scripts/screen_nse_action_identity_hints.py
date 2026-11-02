"""Produce a non-authoritative ticker/ISIN review queue from NSE action data.

NSE's action feed can retrospectively show a current ticker on an older event.
It is therefore useful for discovering candidate ISINs, but is insufficient to
prove a historical ticker mapping or continuity through a reorganisation. This
script deliberately emits review hints, never an identity crosswalk accepted
by ``apply_identity_crosswalk.py``.
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

from src.research.reproducibility import sha256_file


REQUIRED_ACTION_COLUMNS = ("symbol", "isin", "ex_date", "source_record_id")
OUTPUT_COLUMNS = (
    "source_symbol",
    "action_api_first_ex_date",
    "action_api_last_ex_date",
    "action_api_event_count",
    "action_api_distinct_isins",
    "action_api_isins",
    "candidate_asset_id",
    "identity_status",
    "review_note",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-ledger", required=True)
    parser.add_argument("--symbols", required=True, help="CSV whose unique ticker values form the review queue")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--output", required=True, help="New review-hints CSV path")
    args = parser.parse_args()

    action_path = Path(args.action_ledger)
    symbol_path = Path(args.symbols)
    output = Path(args.output)
    if output.exists() or Path(f"{output}.source.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output or sidecar: {output}")
    actions = pd.read_csv(action_path)
    missing = [column for column in REQUIRED_ACTION_COLUMNS if column not in actions.columns]
    if missing:
        raise ValueError(f"Action ledger is missing columns: {missing}")
    source = pd.read_csv(symbol_path)
    if args.symbol_column not in source.columns:
        raise ValueError(f"{symbol_path} is missing {args.symbol_column!r}")
    symbols = (
        source[args.symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not symbols:
        raise ValueError("No non-empty source symbols supplied")
    actions = actions.loc[:, REQUIRED_ACTION_COLUMNS].copy()
    actions["symbol"] = actions["symbol"].astype(str).str.strip().str.upper()
    actions["isin"] = actions["isin"].astype(str).str.strip()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="raise").dt.normalize()
    if actions[["symbol", "isin", "source_record_id"]].eq("").any().any():
        raise ValueError("Action ledger contains an empty required identity field")
    summaries = (
        actions.loc[actions["symbol"].isin(symbols)]
        .groupby("symbol", sort=True)
        .agg(
            action_api_first_ex_date=("ex_date", "min"),
            action_api_last_ex_date=("ex_date", "max"),
            action_api_event_count=("source_record_id", "nunique"),
            action_api_distinct_isins=("isin", "nunique"),
            action_api_isins=("isin", lambda values: ";".join(sorted(values.unique()))),
        )
    )
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        if symbol not in summaries.index:
            rows.append(
                {
                    "source_symbol": symbol,
                    "action_api_first_ex_date": pd.NaT,
                    "action_api_last_ex_date": pd.NaT,
                    "action_api_event_count": 0,
                    "action_api_distinct_isins": 0,
                    "action_api_isins": "",
                    "candidate_asset_id": "",
                    "identity_status": "unobserved_in_action_ledger",
                    "review_note": "No action event; action feed supplies no ISIN hint for this ticker.",
                }
            )
            continue
        summary = summaries.loc[symbol]
        distinct_isins = int(summary["action_api_distinct_isins"])
        rows.append(
            {
                "source_symbol": symbol,
                "action_api_first_ex_date": summary["action_api_first_ex_date"],
                "action_api_last_ex_date": summary["action_api_last_ex_date"],
                "action_api_event_count": int(summary["action_api_event_count"]),
                "action_api_distinct_isins": distinct_isins,
                "action_api_isins": summary["action_api_isins"],
                "candidate_asset_id": summary["action_api_isins"] if distinct_isins == 1 else "",
                "identity_status": "candidate_not_verified" if distinct_isins == 1 else "multiple_isins_require_review",
                "review_note": (
                    "Action feed is an ISIN hint only; verify time-bounded provider-symbol mapping and corporate-action continuity."
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    review = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    review.to_csv(output, index=False)
    metadata = {
        "kind": "nse-corporate-actions-identity-review-hints",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_ledger": str(action_path),
        "action_ledger_sha256": sha256_file(action_path),
        "action_ledger_sidecar": str(Path(f"{action_path}.source.json")),
        "action_ledger_sidecar_sha256": sha256_file(Path(f"{action_path}.source.json")),
        "source_symbols": str(symbol_path),
        "source_symbols_sha256": sha256_file(symbol_path),
        "source_symbol_column": args.symbol_column,
        "symbols": len(symbols),
        "candidate_not_verified": int((review["identity_status"] == "candidate_not_verified").sum()),
        "multiple_isins_require_review": int((review["identity_status"] == "multiple_isins_require_review").sum()),
        "unobserved_in_action_ledger": int((review["identity_status"] == "unobserved_in_action_ledger").sum()),
        "crosswalk_eligibility": "not_eligible_without_time_bounded_external_identity_evidence",
    }
    Path(f"{output}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(review)} non-authoritative NSE identity hints to {output}")


if __name__ == "__main__":
    main()
