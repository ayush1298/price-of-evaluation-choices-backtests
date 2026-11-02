"""Merge complete NSE corporate-action review batches with provenance intact.

Inputs must each be all-or-nothing outputs from
``probe_nse_corporate_actions.py``.  The merge validates their sidecars,
rejects overlapping requested date windows, collapses only identical replayed
source records, and writes a new review-only ledger plus source manifest.
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

from src.research.data import reconcile_nse_corporate_action_records
from src.research.reproducibility import sha256_file


EXPECTED_KIND = "official-nse-equity-corporate-actions-review-ledger"


def _read_batch(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    sidecar_path = Path(f"{path}.source.json")
    if not path.exists() or not sidecar_path.exists():
        raise FileNotFoundError(f"Require both ledger and sidecar: {path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != EXPECTED_KIND:
        raise ValueError(f"Unexpected ledger kind in {sidecar_path}: {metadata.get('kind')}")
    if metadata.get("price_adjustment_status") != "unreviewed_events_not_price_adjustments":
        raise ValueError(f"Unexpected price-adjustment status in {sidecar_path}")
    return pd.read_csv(path), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Complete batch CSVs in any order")
    parser.add_argument("--output", required=True, help="New merged review-ledger CSV path")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() or Path(f"{output}.source.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing output or sidecar: {output}")
    batches = [(Path(value), *_read_batch(Path(value))) for value in args.inputs]
    windows = sorted(
        (
            pd.Timestamp(metadata["requested_start"]).normalize(),
            pd.Timestamp(metadata["requested_end"]).normalize(),
            path,
        )
        for path, _, metadata in batches
    )
    for (_, prior_end, prior_path), (current_start, _, current_path) in zip(windows, windows[1:]):
        if current_start <= prior_end:
            raise ValueError(f"Overlapping requested action windows: {prior_path} and {current_path}")
    raw = pd.concat([frame for _, frame, _ in batches], ignore_index=True)
    ledger, reconciliation = reconcile_nse_corporate_action_records(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    metadata = {
        "kind": "official-nse-equity-corporate-actions-review-ledger",
        "source": "Merged NSE corporate actions API batches",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": windows[0][0].date().isoformat(),
        "requested_end": windows[-1][1].date().isoformat(),
        "inputs": [
            {
                "ledger": str(path),
                "ledger_sha256": sha256_file(path),
                "sidecar": str(Path(f"{path}.source.json")),
                "sidecar_sha256": sha256_file(Path(f"{path}.source.json")),
                "rows": metadata.get("rows"),
                "raw_monthly_fetches": metadata.get("monthly_fetches"),
            }
            for path, _, metadata in batches
        ],
        "input_rows": reconciliation.input_records,
        "rows": reconciliation.unique_records,
        "duplicate_records_collapsed": reconciliation.duplicate_records_collapsed,
        "duplicate_policy": "collapse_only_identical_normalised_source_records",
        "price_adjustment_status": "unreviewed_events_not_price_adjustments",
        "paper_eligibility": "requires_action_and_identity_review_before_use_in_adjusted_panel",
    }
    Path(f"{output}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(batches)} complete action batches into {len(ledger)} review-only records at {output}")


if __name__ == "__main__":
    main()
