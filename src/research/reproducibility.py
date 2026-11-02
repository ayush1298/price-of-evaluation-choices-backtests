"""Metadata writers for canonical research runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# A canonical manifest means that a run is reproducible.  It does *not* mean
# that the data, design, or evidence are ready for a paper table.  Paper
# admission is deliberately an explicit opt-in with recorded evidence; the
# default keeps exploratory and unclassified work out of publication exports.
PAPER_EVIDENCE_REQUIREMENTS = (
    "universe_contract_verified",
    "price_adjustment_verified",
    "evaluation_precommitted",
    "independent_confirmation",
    "robustness_matrix_completed",
)


def research_eligibility_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a conservative paper-release classification from a config object.

    A ``paper_eligible`` assertion alone is intentionally insufficient.  A
    submitting researcher must explicitly attest to each requirement in a
    ``paper_evidence`` object.  The fields are manifest metadata, not a
    substitute for auditing the referenced source files; their role is to make
    a missing data/protocol gate mechanically visible and to prevent an
    accidental table export from silently treating a canonical run as a paper
    result.
    """
    scope = config.get("study_scope", "unclassified_nonpaper")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("study_scope must be a non-empty string when supplied")
    declared = config.get("paper_eligible", False)
    if not isinstance(declared, bool):
        raise ValueError("paper_eligible must be a boolean when supplied")
    raw_evidence = config.get("paper_evidence", {})
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("paper_evidence must be an object when supplied")
    evidence = {
        requirement: raw_evidence.get(requirement) is True
        for requirement in PAPER_EVIDENCE_REQUIREMENTS
    }
    missing = [requirement for requirement, satisfied in evidence.items() if not satisfied]
    admitted = declared and not missing
    return {
        "study_scope": scope,
        "declared_paper_eligible": declared,
        "paper_eligible": admitted,
        "paper_evidence": evidence,
        "missing_paper_evidence": missing,
        "release_status": "paper_ready" if admitted else "nonpaper",
    }


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths_with_existing_source_sidecars(input_paths: list[str | Path]) -> list[str | Path]:
    """Include adjacent ``.source.json`` provenance files when they exist.

    Canonical results always hash the direct input CSVs.  Public and vendor
    download workflows additionally write a sidecar that records the retrieval
    source, adjustment convention, or membership construction rule.  Carrying
    that sidecar into a run manifest makes the data lineage reviewable without
    requiring every caller to remember an extra CLI argument.
    """
    expanded: list[str | Path] = []
    seen: set[Path] = set()
    for value in input_paths:
        path = Path(value)
        normalized = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if normalized not in seen:
            expanded.append(value)
            seen.add(normalized)
        sidecar = Path(f"{normalized}.source.json")
        if sidecar.exists() and sidecar not in seen:
            expanded.append(sidecar)
            seen.add(sidecar)
    return expanded


def _display_path(path: Path) -> str:
    """Use a project-relative path where possible, otherwise preserve its absolute path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_run_manifest(
    run_id: str,
    *,
    config_path: str | Path,
    input_paths: list[str | Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write immutable provenance metadata for a canonical run directory."""
    if not run_id or any(character in run_id for character in "/\\"):
        raise ValueError("run_id must be a non-empty simple directory name")
    config = Path(config_path)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    config = config.resolve()
    if not config.exists():
        raise FileNotFoundError(config)
    try:
        config_payload = json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Canonical run config must be a JSON object: {config}") from error
    if not isinstance(config_payload, dict):
        raise ValueError(f"Canonical run config must be a JSON object: {config}")
    eligibility = research_eligibility_from_config(config_payload)
    inputs = [
        (Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path).resolve()
        for path in input_paths
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest inputs do not exist: {missing}")

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    run_directory = PROJECT_ROOT / "results" / "canonical" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "canonical-run-manifest",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "config": {"path": _display_path(config), "sha256": sha256_file(config)},
        "inputs": [
            {"path": _display_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "extra": {
            **(extra or {}),
            # Write the classification after runner-specific metadata so no
            # caller can accidentally override the central release gate.
            "study_scope": eligibility["study_scope"],
            "paper_eligible": eligibility["paper_eligible"],
            "paper_eligibility": eligibility,
        },
    }
    manifest_path = run_directory / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
