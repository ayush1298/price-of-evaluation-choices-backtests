"""Create a fully observed shared-calendar panel for a declared fixed universe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import synchronise_fixed_universe_panel, validate_price_panel
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", help="Optional inclusive first trading date for the synchronized study panel")
    parser.add_argument("--end", help="Optional inclusive final trading date for the synchronized study panel")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prices_path = _project_path(args.prices)
    symbols_path = _project_path(args.symbols)
    output_path = _project_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; review it and pass --overwrite if intended")
    manifest = pd.read_csv(symbols_path)
    if args.symbol_column not in manifest.columns:
        raise ValueError(f"Symbol manifest is missing {args.symbol_column!r}")
    source, input_report = validate_price_panel(pd.read_csv(prices_path), require_ohlcv=True)
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    if start is not None and end is not None and start > end:
        raise ValueError("--start cannot be after --end")
    if start is not None:
        source = source.loc[source["date"] >= start].copy()
    if end is not None:
        source = source.loc[source["date"] <= end].copy()
    if source.empty:
        raise ValueError("No source observations remain after the requested date bounds")
    source, source_report = validate_price_panel(source, require_ohlcv=True)
    synchronized = synchronise_fixed_universe_panel(source, manifest[args.symbol_column], require_ohlcv=True)
    _, synchronized_report = validate_price_panel(synchronized, require_ohlcv=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synchronized.to_csv(output_path, index=False)
    output_path.with_suffix(output_path.suffix + ".source.json").write_text(
        json.dumps(
            {
                "kind": "synchronized-fixed-universe-panel",
                "source_prices_path": str(prices_path),
                "source_prices_sha256": sha256_file(prices_path),
                "symbols_path": str(symbols_path),
                "symbols_sha256": sha256_file(symbols_path),
                "rule": "retain only dates with all declared assets observed; no imputation or forward fill",
                "input_rows": input_report.n_rows,
                "source_rows": source_report.n_rows,
                "source_dates": int(source["date"].nunique()),
                "date_bounds": {
                    "start": start.isoformat() if start is not None else None,
                    "end": end.isoformat() if end is not None else None,
                },
                "retained_rows": synchronized_report.n_rows,
                "retained_dates": int(synchronized["date"].nunique()),
                "removed_dates": int(source["date"].nunique() - synchronized["date"].nunique()),
                "n_assets": synchronized_report.n_assets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {synchronized_report.n_rows:,} rows across {synchronized['date'].nunique():,} "
        f"fully observed dates to {output_path}"
    )


if __name__ == "__main__":
    main()
