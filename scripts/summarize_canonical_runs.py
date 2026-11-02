"""Create a publication table from completed canonical run directories only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.evaluation import canonical_paper_table, collect_canonical_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="results/canonical")
    parser.add_argument("--output", required=True, help="CSV path for the publication-table draft")
    parser.add_argument(
        "--include-nonpaper",
        action="store_true",
        help="Explicitly include exploratory/unclassified runs for an internal diagnostic table",
    )
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    table = canonical_paper_table(collect_canonical_runs(runs_root, include_exploratory=args.include_nonpaper))
    if table.empty:
        raise ValueError(
            "No explicitly paper-eligible manifest-backed runs were found. "
            "Use --include-nonpaper only for an internal diagnostic table."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    table_kind = "internal diagnostic" if args.include_nonpaper else "paper-ready"
    try:
        display_output = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_output = output
    print(f"Wrote {len(table)} {table_kind} canonical run rows to {display_output}")


if __name__ == "__main__":
    main()
