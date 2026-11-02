"""Apply evidence-linked split/bonus price factors to a raw OHLCV panel.

This intentionally does not infer actions from price jumps and refuses
dividends, mergers, spin-offs, and delistings.  Its output is price-scale
adjusted only, not a total-return panel or a paper-ready dataset by itself.
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

from src.research.data import apply_price_scale_actions
from src.research.reproducibility import input_paths_with_existing_source_sidecars, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, help="Raw genuine-OHLCV CSV")
    parser.add_argument("--actions", required=True, help="Reviewed split/bonus adjustment event CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prices_path = Path(args.prices)
    actions_path = Path(args.actions)
    output_path = Path(args.output)
    if not prices_path.exists() or not actions_path.exists():
        missing = [str(path) for path in (prices_path, actions_path) if not path.exists()]
        raise FileNotFoundError(f"Missing input file(s): {missing}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; review it and pass --overwrite if intended")
    adjusted, audit = apply_price_scale_actions(pd.read_csv(prices_path), pd.read_csv(actions_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(output_path, index=False)
    inputs = input_paths_with_existing_source_sidecars([prices_path, actions_path])
    metadata = {
        "kind": "price-scale-adjusted-ohlcv",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": [str(path) for path in inputs],
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "n_actions": audit.n_actions,
        "n_symbols_with_actions": audit.n_symbols,
        "affected_price_rows": audit.affected_price_rows,
        "action_types": list(audit.action_types),
        "adjustment_scope": audit.adjustment_scope,
        "not_a_total_return_series": True,
        "unmodeled_event_types": ["cash_dividend", "merger", "spin_off", "rights_issue", "delisting"],
        "paper_eligible_by_itself": False,
        "warning": (
            "This output may not be merged with a provider-adjusted panel or used as a paper-ready return series "
            "until corporate-action completeness, identity mapping, and delisting treatment are independently verified."
        ),
    }
    Path(f"{output_path}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(adjusted):,} price-scale-adjusted OHLCV rows to {output_path}")


if __name__ == "__main__":
    main()
