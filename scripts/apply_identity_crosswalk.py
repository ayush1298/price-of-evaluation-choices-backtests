"""Map dated membership and provider prices to audited stable asset IDs.

Only ``verification_status=verified`` mappings are accepted.  A user may
retain unmapped literal symbols for an explicitly exploratory build, but the
sidecar records that exception and the output never becomes paper-ready by
itself.
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

from src.research.data import map_membership_to_asset_ids, map_price_panel_to_asset_ids
from src.research.reproducibility import input_paths_with_existing_source_sidecars, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--output-prices", required=True)
    parser.add_argument("--output-membership", required=True)
    parser.add_argument(
        "--allow-unmapped-literal-symbols",
        action="store_true",
        help="Keep unlisted identity mappings only for a labelled exploratory build",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = {name: Path(getattr(args, name)) for name in ("prices", "membership", "crosswalk")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")
    output_prices = Path(args.output_prices)
    output_membership = Path(args.output_membership)
    if output_prices == output_membership:
        raise ValueError("Price and membership outputs must be distinct files")
    existing = [path for path in (output_prices, output_membership) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {existing}; review target(s) and pass --overwrite if intended")

    crosswalk = pd.read_csv(paths["crosswalk"])
    mapped_prices, price_audit = map_price_panel_to_asset_ids(
        pd.read_csv(paths["prices"]),
        crosswalk,
        allow_unmapped_literal_symbols=args.allow_unmapped_literal_symbols,
    )
    mapped_membership, membership_audit = map_membership_to_asset_ids(
        pd.read_csv(paths["membership"]),
        crosswalk,
        allow_unmapped_literal_symbols=args.allow_unmapped_literal_symbols,
    )
    output_prices.parent.mkdir(parents=True, exist_ok=True)
    output_membership.parent.mkdir(parents=True, exist_ok=True)
    mapped_prices.to_csv(output_prices, index=False)
    mapped_membership.to_csv(output_membership, index=False)

    inputs = input_paths_with_existing_source_sidecars(paths.values())
    common = {
        "kind": "time_bounded_identity_crosswalk_application",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": [str(path) for path in inputs],
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "crosswalk_requires_verified_rows": True,
        "unmapped_literal_symbols_allowed": args.allow_unmapped_literal_symbols,
        "paper_eligible_by_itself": False,
        "warning": (
            "Identity mapping does not establish corporate-action adjustment, delisting treatment, or independently "
            "verified membership history. It must be combined with those audited layers before any paper claim."
        ),
    }
    price_metadata = {
        **common,
        "output_role": "prices",
        "mapped_rows": price_audit.mapped_rows,
        "retained_literal_rows": price_audit.retained_literal_rows,
        "mapped_source_symbols": list(price_audit.mapped_source_symbols),
    }
    membership_metadata = {
        **common,
        "output_role": "membership",
        "mapped_rows": membership_audit.mapped_rows,
        "retained_literal_rows": membership_audit.retained_literal_rows,
        "mapped_source_symbols": list(membership_audit.mapped_source_symbols),
    }
    Path(f"{output_prices}.source.json").write_text(json.dumps(price_metadata, indent=2) + "\n", encoding="utf-8")
    Path(f"{output_membership}.source.json").write_text(
        json.dumps(membership_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(mapped_prices):,} price rows and {len(mapped_membership):,} membership rows "
        "with time-bounded identity mapping"
    )


if __name__ == "__main__":
    main()
