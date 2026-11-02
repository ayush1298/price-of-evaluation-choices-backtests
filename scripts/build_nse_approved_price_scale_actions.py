"""Materialise explicit price-scale actions from a fully reviewed NSE queue.

This command refuses incomplete decisions and never turns the parser's proposed
factor into an adjustment by itself.  The output is the only schema accepted
by ``adjust_price_scale_actions.py`` / ``apply_price_scale_actions``.
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

from src.research.data import materialise_approved_price_scale_actions
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, help="Split/bonus review queue produced by screen_nse_price_scale_actions.py")
    parser.add_argument("--decisions", required=True, help="Human-reviewed decision CSV using the versioned template")
    parser.add_argument("--output", required=True, help="Output explicit adjustment CSV")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    review_path = _project_path(args.review)
    decision_path = _project_path(args.decisions)
    output_path = _project_path(args.output)
    missing = [str(path) for path in (review_path, decision_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")
    sidecar_path = Path(f"{output_path}.source.json")
    if not args.overwrite and (output_path.exists() or sidecar_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing output or sidecar: {output_path}")

    actions, audit = materialise_approved_price_scale_actions(
        pd.read_csv(review_path),
        pd.read_csv(decision_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions.to_csv(output_path, index=False)
    metadata = {
        "kind": "nse-human-approved-price-scale-actions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "decision_path": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "audit": {
            "review_candidates": audit.review_candidates,
            "decision_rows": audit.decision_rows,
            "all_candidates_resolved": audit.all_candidates_resolved,
            "approved_price_scale_factors": audit.approved_price_scale_factors,
            "excluded_from_raw_panel": audit.excluded_from_raw_panel,
            "requires_total_return_source": audit.requires_total_return_source,
            "approvals_with_factor_override": audit.approvals_with_factor_override,
        },
        "warning": (
            "Approved factors adjust split/bonus price scale only. This output does not establish dividend total "
            "returns, merger/demerger consideration, delisting treatment, or a paper-eligible panel."
        ),
    }
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(actions)} human-approved price-scale actions to {output_path}")


if __name__ == "__main__":
    main()
