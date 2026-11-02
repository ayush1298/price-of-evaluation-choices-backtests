"""Download genuine adjusted OHLCV for prototyping; not a universe-history feed."""

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
import yfinance

from src.research.data import download_yfinance_ohlcv_audited, validate_price_panel
from src.research.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="CSV containing one provider ticker column")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument(
        "--output-symbol-column",
        help="Optional CSV column whose values replace provider tickers in the saved canonical panel",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument(
        "--allow-non-tradable-price-series",
        action="store_true",
        help="Permit index-style series with constant/zero volume; never use these as tradeable universe assets",
    )
    parser.add_argument(
        "--allow-missing-symbols",
        action="store_true",
        help="Save a partial panel only after recording every requested ticker Yahoo omitted",
    )
    parser.add_argument(
        "--drop-symbols-with-invalid-ohlcv",
        action="store_true",
        help="Exclude and record symbols with materially inconsistent provider OHLCV; never repair bars",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite after reviewing the target")
    symbols = pd.read_csv(args.symbols)
    if args.symbol_column not in symbols.columns:
        raise ValueError(f"Ticker file is missing column {args.symbol_column!r}")
    if args.output_symbol_column is not None and args.output_symbol_column not in symbols.columns:
        raise ValueError(f"Ticker file is missing {args.output_symbol_column!r}")
    provider_symbols = symbols[args.symbol_column].astype(str).str.strip()
    if provider_symbols.duplicated().any():
        raise ValueError("Provider ticker column contains duplicates")
    if args.output_symbol_column is None:
        output_symbols = provider_symbols
    else:
        output_symbols = symbols[args.output_symbol_column].astype(str).str.strip()
        if (output_symbols == "").any() or output_symbols.duplicated().any():
            raise ValueError("Output symbol column must contain unique non-empty identifiers")
    audit = download_yfinance_ohlcv_audited(
        provider_symbols,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        pause_seconds=args.pause_seconds,
        require_ohlcv=not args.allow_non_tradable_price_series,
    )
    if audit.missing_symbols and not args.allow_missing_symbols:
        raise ValueError(
            f"Yahoo Finance omitted {len(audit.missing_symbols)} requested symbols: {list(audit.missing_symbols)}. "
            "Pass --allow-missing-symbols only for an explicitly partial exploratory panel."
        )
    if audit.invalid_symbols and not args.drop_symbols_with_invalid_ohlcv:
        raise ValueError(
            "Yahoo Finance returned materially invalid OHLCV for "
            f"{sorted(audit.invalid_symbols)}. Pass --drop-symbols-with-invalid-ohlcv only to exclude them."
        )
    panel = audit.panel.copy()
    symbol_mapping = dict(zip(provider_symbols, output_symbols, strict=True))
    panel["symbol"] = panel["symbol"].map(symbol_mapping)
    if panel["symbol"].isna().any():
        raise ValueError("Downloaded provider ticker is absent from the requested symbol mapping")
    if panel.empty:
        raise ValueError("No OHLCV observations were downloaded")
    _, report = validate_price_panel(panel, require_ohlcv=not args.allow_non_tradable_price_series)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    source_path = output.with_suffix(output.suffix + ".source.json")
    source_path.write_text(
        json.dumps(
            {
                "kind": "public-prototype-ohlcv-source",
                "provider": "yfinance",
                "provider_library_version": yfinance.__version__,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "auto_adjust": True,
                "require_ohlcv": not args.allow_non_tradable_price_series,
                "symbols_file": str(args.symbols),
                "symbols_file_sha256": sha256_file(args.symbols),
                "provider_symbol_column": args.symbol_column,
                "output_symbol_column": args.output_symbol_column or args.symbol_column,
                "requested_symbols": list(audit.requested_symbols),
                "downloaded_symbols": list(audit.downloaded_symbols),
                "missing_symbols": list(audit.missing_symbols),
                "invalid_ohlcv_symbols": audit.invalid_symbols,
                "missing_symbol_policy": "allow_and_record" if args.allow_missing_symbols else "reject",
                "invalid_ohlcv_policy": (
                    "drop_symbol_and_record" if args.drop_symbols_with_invalid_ohlcv else "reject"
                ),
                "start": args.start,
                "end": args.end,
                "output": str(output),
                "n_assets": report.n_assets,
                "n_rows": report.n_rows,
                "warning": "Prices do not provide point-in-time constituent history and are not paper-ready alone.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {report.n_rows:,} genuine OHLCV rows for {report.n_assets} assets to {output}")


if __name__ == "__main__":
    main()
