"""Validate a versioned research data contract and emit its provenance report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.data.contract import load_data_contract, validate_data_contract
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="Versioned JSON data contract")
    parser.add_argument("--output", help="Optional validation-report JSON path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contract_path = _project_path(args.contract)
    contract = load_data_contract(contract_path)
    report = validate_data_contract(contract, project_root=PROJECT_ROOT)
    report["contract_file"] = {
        "path": str(contract_path.relative_to(PROJECT_ROOT))
        if contract_path.is_relative_to(PROJECT_ROOT)
        else str(contract_path),
        "sha256": sha256_file(contract_path),
    }
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = _project_path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Validated {report['contract']['contract_id']} and wrote {output}")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
