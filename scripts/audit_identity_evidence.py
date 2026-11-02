"""Validate the evidence ledger attached to a historical identity crosswalk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data.identity_evidence import audit_identity_evidence
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", help="Optional JSON audit report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    crosswalk_path = _project_path(args.crosswalk)
    evidence_path = _project_path(args.evidence)
    audit = audit_identity_evidence(pd.read_csv(crosswalk_path), pd.read_csv(evidence_path))
    report = {
        "kind": "historical-identity-evidence-audit",
        **audit.as_jsonable(),
        "inputs": {
            "crosswalk": {"path": str(crosswalk_path), "sha256": sha256_file(crosswalk_path)},
            "evidence": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
        },
        "warning": (
            "This audit only checks identity-evidence coverage. It does not establish corporate-action adjustment, "
            "delisting treatment, point-in-time membership, or paper eligibility."
        ),
    }
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = _project_path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Wrote identity evidence audit to {output}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
