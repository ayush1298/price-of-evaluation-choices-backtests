"""Import a community Nifty 50 historical-weight panel as exploratory membership.

The source's positive weights are transformed into dated eligibility snapshots.
It remains a community reconstruction until independently reconciled to the
official Nifty Indices reports and paired with auditable price/ticker history.
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

from src.research.data import membership_from_kaggle_nifty50_weights
from src.research.reproducibility import sha256_file


DEFAULT_SOURCE_URL = "https://www.kaggle.com/datasets/megachonk/historical-nifty-50-constituent-weights-20y"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Downloaded Kaggle weights.csv")
    parser.add_argument("--output", required=True, help="Long-format output membership CSV")
    parser.add_argument("--dataset-version", required=True, type=int)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; review it and pass --overwrite if intended")
    membership = membership_from_kaggle_nifty50_weights(pd.read_csv(weights_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_path, index=False)
    metadata = {
        "kind": "community-kaggle-nifty50-point-in-time-membership-candidate",
        "source": "Kaggle megachonk/historical-nifty-50-constituent-weights-20y",
        "source_url": args.source_url,
        "dataset_version": args.dataset_version,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_weights_path": str(weights_path),
        "raw_weights_sha256": sha256_file(weights_path),
        "snapshot_semantics": "positive source weight denotes eligible; latest snapshot is carried forward by the canonical lookup",
        "source_frequency": "monthly_and_event_driven_community_snapshots",
        "universe_scope": "historical_nifty50_membership_candidate_pending_official_reconciliation",
        "eligible_for_canonical_historical_index_claim": False,
        "rows": int(len(membership)),
        "snapshots": int(membership["date"].nunique()),
        "symbols": int(membership["symbol"].nunique()),
        "start": membership["date"].min().isoformat(),
        "end": membership["date"].max().isoformat(),
    }
    Path(f"{output_path}.source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(membership):,} Nifty 50 membership rows to {output_path}")


if __name__ == "__main__":
    main()
