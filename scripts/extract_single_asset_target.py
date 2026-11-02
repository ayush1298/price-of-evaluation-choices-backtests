"""Extract one genuine benchmark OHLCV series from a canonical public panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.research.data import validate_price_panel
from src.research.reproducibility import sha256_file


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prices_path = _project_path(args.prices)
    output_path = _project_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite after reviewing it")
    panel, _ = validate_price_panel(pd.read_csv(prices_path), require_ohlcv=True)
    target = panel.loc[panel["symbol"] == args.symbol, ["date", "open", "high", "low", "close", "volume"]].copy()
    if target.empty:
        raise ValueError(f"Symbol {args.symbol!r} is absent from {prices_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(output_path, index=False)
    Path(f"{output_path}.source.json").write_text(
        json.dumps(
            {
                "kind": "single-asset-target-extract",
                "symbol": args.symbol,
                "input_prices_path": str(prices_path),
                "input_prices_sha256": sha256_file(prices_path),
                "n_rows": len(target),
                "start": str(target["date"].min()),
                "end": str(target["date"].max()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(target):,} {args.symbol} target rows to {output_path}")


if __name__ == "__main__":
    main()
