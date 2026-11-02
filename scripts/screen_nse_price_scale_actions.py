"""Create a manual-review queue for possible NSE split/bonus adjustments.

This command never writes the adjustment-table schema accepted by
``adjust_price_scale_actions.py``. Its factors are proposals from conventional
text patterns only and must be independently reviewed with the linked source.
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

from src.research.data import extract_price_scale_review_candidates
from src.research.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols-csv", help="Optional CSV limiting candidates to a symbol column")
    parser.add_argument("--symbol-column", default="symbol")
    args = parser.parse_args()

    action_path = Path(args.action_ledger)
    output = Path(args.output)
    if output.exists() or Path(f"{output}.source.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output or sidecar: {output}")
    symbols: tuple[str, ...] | None = None
    if args.symbols_csv:
        symbol_path = Path(args.symbols_csv)
        source = pd.read_csv(symbol_path)
        if args.symbol_column not in source.columns:
            raise ValueError(f"{symbol_path} is missing {args.symbol_column!r}")
        symbols = tuple(source[args.symbol_column].dropna().astype(str).str.strip().str.upper().unique())
    review, audit = extract_price_scale_review_candidates(pd.read_csv(action_path), symbols=symbols)
    output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output, index=False)
    metadata = {
        "kind": "nse-corporate-action-price-scale-review-queue",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "action_ledger": str(action_path),
        "action_ledger_sha256": sha256_file(action_path),
        "action_ledger_sidecar": str(Path(f"{action_path}.source.json")),
        "action_ledger_sidecar_sha256": sha256_file(Path(f"{action_path}.source.json")),
        "symbol_filter": None if symbols is None else list(symbols),
        "input_events": audit.input_events,
        "matched_events": audit.matched_events,
        "requires_human_approval": audit.simple_candidates,
        "blocked_complex_action": audit.blocked_complex_events,
        "unparsed_price_scale_text": audit.unparsed_events,
        "adjustment_status": "proposed_factors_not_applied",
        "next_step": "independently verify each approved event before creating the explicit price-scale action table",
    }
    Path(f"{output}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(review)} proposed-action review rows to {output}")


if __name__ == "__main__":
    main()
