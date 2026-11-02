"""Download the public Fama-French daily five-factor archive with provenance."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.data import parse_fama_french_daily_factors
from src.research.reproducibility import sha256_file


DEFAULT_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output decimal-return daily-factor CSV")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite after reviewing the target")

    request = urllib.request.Request(args.url, headers={"User-Agent": "research-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        archive = response.read()
    factors = parse_fama_french_daily_factors(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(output, index=False)
    sidecar = output.with_suffix(output.suffix + ".source.json")
    sidecar.write_text(
        json.dumps(
            {
                "kind": "public-academic-factor-source",
                "provider": "Kenneth French Data Library",
                "url": args.url,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": _sha256_bytes(archive),
                "output": str(output),
                "output_sha256": sha256_file(output),
                "return_units": "decimal daily returns; source percentages divided by 100",
                "factor_columns": [column for column in factors.columns if column != "date"],
                "start": factors["date"].min().isoformat(),
                "end": factors["date"].max().isoformat(),
                "n_observations": len(factors),
                "warning": "Academic factor data are for ex-post evaluation, not live-trading features.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(factors):,} daily factor observations to {output}")


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
