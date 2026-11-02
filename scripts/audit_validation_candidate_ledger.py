"""Audit a predeclared validation-candidate ledger against canonical runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evaluation import audit_validation_candidate_ledger, load_candidate_ledger
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument("--output", help="Optional JSON audit report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ledger_path = _project_path(args.ledger)
    results_root = _project_path(args.results_root)
    audit = audit_validation_candidate_ledger(
        load_candidate_ledger(ledger_path),
        results_root=results_root,
    )
    payload = {
        "kind": "validation-candidate-ledger-audit",
        "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "results_root": str(results_root),
        **audit.as_jsonable(),
    }
    content = json.dumps(payload, indent=2) + "\n"
    if args.output:
        output = _project_path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Wrote validation-candidate ledger audit to {output}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
