"""Build causal daily regime covariates from a public non-tradable price series."""

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

from src.research.regimes import build_price_regime_covariates
from src.research.reproducibility import sha256_file


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lookback-sessions", type=int, default=21)
    parser.add_argument("--annualization-factor", type=int, default=252)
    args = parser.parse_args()
    prices_path = _project_path(args.prices)
    output_path = _project_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    covariates = build_price_regime_covariates(
        pd.read_csv(prices_path),
        symbol=args.symbol,
        prefix=args.prefix,
        lookback_sessions=args.lookback_sessions,
        annualization_factor=args.annualization_factor,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(output_path, index=False)
    output_path.with_suffix(output_path.suffix + ".source.json").write_text(
        json.dumps(
            {
                "kind": "public-price-regime-covariates",
                "input_prices": str(prices_path),
                "input_prices_sha256": sha256_file(prices_path),
                "symbol": args.symbol,
                "prefix": args.prefix,
                "lookback_sessions": args.lookback_sessions,
                "annualization_factor": args.annualization_factor,
                "n_rows": len(covariates),
                "start": pd.Timestamp(covariates["date"].min()).isoformat(),
                "end": pd.Timestamp(covariates["date"].max()).isoformat(),
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "Covariates are observed market-state inputs, not tradeable universe assets.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(covariates):,} causal covariate rows to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
