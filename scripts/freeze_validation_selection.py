"""Freeze a selected candidate from a complete predeclared validation ledger."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evaluation import freeze_validation_selection, load_candidate_ledger
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ledger_path = _project_path(args.ledger)
    output = _project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {output}")
    selection = freeze_validation_selection(
        load_candidate_ledger(ledger_path),
        results_root=_project_path(args.results_root),
    )
    payload = {
        **selection,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Froze validation selection {selection['selected_candidate_id']} to {output}")


if __name__ == "__main__":
    main()
