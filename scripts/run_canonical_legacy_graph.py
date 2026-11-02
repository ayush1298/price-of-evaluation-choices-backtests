"""Generate one isolated, seeded DeepCNL baseline graph.

The command produces a canonical run directory and never writes to the legacy
``outputs/graphs`` location. Use a point-in-time filtered OHLCV input file;
the command rejects close-only and synthetic-constant-volume inputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.graphs import LegacyDeepCNLConfig, run_legacy_deepcnl
from src.research.reproducibility import input_paths_with_existing_source_sidecars, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prices", required=True, help="Real long-format OHLCV CSV")
    parser.add_argument("--target", required=True, help="Market-index target CSV accepted by MTP1 Data_util")
    parser.add_argument("--start", required=True, help="Inclusive training start date")
    parser.add_argument("--end", required=True, help="Inclusive training end date")
    parser.add_argument("--config", default="configs/canonical_us_monthly.json")
    parser.add_argument("--ticker-num", type=int, default=470)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rare-ratio", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run = run_legacy_deepcnl(
        args.prices,
        args.target,
        training_start=args.start,
        training_end=args.end,
        config=LegacyDeepCNLConfig(
            ticker_num=args.ticker_num,
            epochs=args.epochs,
            rare_ratio=args.rare_ratio,
            seed=args.seed,
        ),
    )
    manifest_path = write_run_manifest(
        args.run_id,
        config_path=args.config,
        input_paths=input_paths_with_existing_source_sidecars([args.prices, args.target]),
        extra={"deepcnl": run.metadata},
    )
    output_directory = manifest_path.parent
    with (output_directory / "deepcnl_graph.pkl").open("wb") as stream:
        pickle.dump(run.graph, stream)
    (output_directory / "deepcnl_graph_metadata.json").write_text(
        json.dumps(run.metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved canonical DeepCNL graph to {output_directory.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
