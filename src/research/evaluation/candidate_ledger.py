"""Validation-candidate ledgers that prevent post-hoc outer-test selection.

The ledger is written before a confirmation run and names every candidate that
is allowed to compete.  It binds candidate settings to manifest-backed
validation runs, then either freezes one selection or fails closed.  An
archival ledger may audit older development work, but cannot retroactively
authorise an outer-test selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


LEDGER_KIND = "validation-candidate-ledger"
SCHEMA_VERSION = 1
REQUIRED_LEDGER_FIELDS = (
    "kind",
    "schema_version",
    "ledger_id",
    "study_scope",
    "data_contract_id",
    "selection_permitted",
    "selection_metric",
    "validation_window",
    "candidates",
)
REQUIRED_CANDIDATE_FIELDS = ("candidate_id", "run_id", "expected_manifest_values")


@dataclass(frozen=True)
class CandidateLedgerAudit:
    """Whether a validation ledger is complete enough to freeze a candidate."""

    ledger_id: str
    study_scope: str
    data_contract_id: str
    selection_permitted: bool
    selection_metric: str
    candidate_count: int
    complete_candidate_count: int
    incomplete_candidate_count: int
    selection_status: str
    selected_candidate_id: str | None
    selected_metric_value: float | None
    candidates: tuple[dict[str, Any], ...]

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_candidate_ledger(path: str | Path) -> dict[str, Any]:
    """Read one versioned validation-candidate ledger."""
    ledger_path = Path(path)
    try:
        value = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Candidate ledger must be valid JSON: {ledger_path}") from error
    if not isinstance(value, dict):
        raise ValueError("Candidate ledger must be a JSON object")
    return value


def audit_validation_candidate_ledger(
    ledger: Mapping[str, Any],
    *,
    results_root: str | Path,
) -> CandidateLedgerAudit:
    """Audit candidate manifests and select only from a complete permitted ledger."""
    validated = _validate_ledger(ledger)
    root = Path(results_root)
    outcomes: list[dict[str, Any]] = []
    for candidate in validated["candidates"]:
        outcomes.append(_audit_candidate(candidate, validated, root))
    complete = [outcome for outcome in outcomes if outcome["status"] == "complete"]
    incomplete_count = len(outcomes) - len(complete)
    selected_id: str | None = None
    selected_metric: float | None = None
    if not validated["selection_permitted"]:
        selection_status = "selection_not_permitted_by_ledger"
    elif incomplete_count:
        selection_status = "selection_blocked_incomplete_candidates"
    else:
        winner = max(complete, key=lambda outcome: (float(outcome["metric_value"]), str(outcome["candidate_id"])))
        selected_id = str(winner["candidate_id"])
        selected_metric = float(winner["metric_value"])
        selection_status = "selection_ready_to_freeze"
    return CandidateLedgerAudit(
        ledger_id=validated["ledger_id"],
        study_scope=validated["study_scope"],
        data_contract_id=validated["data_contract_id"],
        selection_permitted=validated["selection_permitted"],
        selection_metric=validated["selection_metric"],
        candidate_count=len(outcomes),
        complete_candidate_count=len(complete),
        incomplete_candidate_count=incomplete_count,
        selection_status=selection_status,
        selected_candidate_id=selected_id,
        selected_metric_value=selected_metric,
        candidates=tuple(outcomes),
    )


def freeze_validation_selection(
    ledger: Mapping[str, Any],
    *,
    results_root: str | Path,
) -> dict[str, Any]:
    """Return an immutable selection record only for a complete permitted ledger.

    The caller must persist this object before opening the outer-test period.
    Its selected validation-run config hash and expected runner values let an
    outer-test runner verify that it did not drift after validation selection.
    """
    validated = _validate_ledger(ledger)
    audit = audit_validation_candidate_ledger(ledger, results_root=results_root)
    if audit.selection_status != "selection_ready_to_freeze" or audit.selected_candidate_id is None:
        raise ValueError(
            "Cannot freeze validation selection unless a permitted ledger has complete matching candidates"
        )
    selected = next(
        candidate for candidate in validated["candidates"] if candidate["candidate_id"] == audit.selected_candidate_id
    )
    manifest_path = Path(results_root) / selected["run_id"] / "run_manifest.json"
    manifest = _load_json_object(manifest_path, "selected validation run manifest")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("sha256"), str):
        raise ValueError("Selected validation manifest has no config SHA-256")
    return {
        "kind": "frozen-validation-selection",
        "schema_version": 1,
        "ledger_id": validated["ledger_id"],
        "source_study_scope": validated["study_scope"],
        "source_data_contract_id": validated["data_contract_id"],
        "validation_window": {
            field: None if timestamp is None else timestamp.date().isoformat()
            for field, timestamp in validated["validation_window"].items()
        },
        "selection_metric": validated["selection_metric"],
        "selected_candidate_id": audit.selected_candidate_id,
        "selected_metric_value": audit.selected_metric_value,
        "selected_run_id": selected["run_id"],
        "selected_expected_manifest_values": selected["expected_manifest_values"],
        "selected_validation_config": dict(config),
    }


def _validate_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_LEDGER_FIELDS if field not in ledger]
    if missing:
        raise ValueError(f"Candidate ledger is missing required fields: {missing}")
    if ledger["kind"] != LEDGER_KIND or ledger["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Candidate ledger must be {LEDGER_KIND!r} schema version {SCHEMA_VERSION}")
    validated: dict[str, Any] = {}
    for field in ("ledger_id", "study_scope", "data_contract_id", "selection_metric"):
        value = ledger[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Candidate ledger {field!r} must be a non-empty string")
        validated[field] = value.strip()
    if not isinstance(ledger["selection_permitted"], bool):
        raise ValueError("Candidate ledger selection_permitted must be a boolean")
    validated["selection_permitted"] = ledger["selection_permitted"]
    window = ledger["validation_window"]
    if not isinstance(window, Mapping):
        raise ValueError("Candidate ledger validation_window must be an object")
    if "first_signal_date" not in window or "last_signal_date" not in window:
        raise ValueError("Candidate ledger validation_window requires first_signal_date and last_signal_date")
    try:
        first = pd.Timestamp(window["first_signal_date"]).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("Candidate ledger validation_window requires a valid first_signal_date") from error
    if first is pd.NaT or pd.isna(first):
        raise ValueError("Candidate ledger validation_window requires a valid first_signal_date")
    # An explicit null end date declares an open-ended window that runs to the
    # end of the validated panel.  A confirmation study cannot always name its
    # final signal in advance, and forcing a placeholder date would make the
    # ledger disagree with the manifests it exists to check.  Null is still a
    # commitment: the runs must record an open end too, so the ledger and the
    # runs must agree either way.
    raw_last = window["last_signal_date"]
    if raw_last is None:
        last = None
    else:
        try:
            last = pd.Timestamp(raw_last).normalize()
        except (TypeError, ValueError) as error:
            raise ValueError("Candidate ledger validation_window requires a valid last_signal_date") from error
        if last is pd.NaT or pd.isna(last):
            raise ValueError("Candidate ledger validation_window requires a valid last_signal_date")
        if last < first:
            raise ValueError("Candidate ledger validation window ends before it starts")
    validated["validation_window"] = {"first_signal_date": first, "last_signal_date": last}
    candidates = ledger["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate ledger candidates must be a non-empty list")
    ids: set[str] = set()
    validated_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Each candidate-ledger entry must be an object")
        missing_candidate = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in candidate]
        if missing_candidate:
            raise ValueError(f"Candidate ledger entry is missing fields: {missing_candidate}")
        candidate_id = candidate["candidate_id"]
        run_id = candidate["run_id"]
        expected = candidate["expected_manifest_values"]
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Candidate IDs must be non-empty strings")
        if candidate_id in ids:
            raise ValueError(f"Candidate IDs must be unique; repeated {candidate_id!r}")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Candidate run IDs must be non-empty strings")
        if not isinstance(expected, Mapping):
            raise ValueError("Candidate expected_manifest_values must be an object")
        if not all(isinstance(path, str) and path.startswith("extra.") for path in expected):
            raise ValueError("Candidate expected manifest keys must be non-empty paths beginning 'extra.'")
        ids.add(candidate_id)
        validated_candidates.append(
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "expected_manifest_values": dict(expected),
            }
        )
    validated["candidates"] = validated_candidates
    return validated


def _audit_candidate(candidate: Mapping[str, Any], ledger: Mapping[str, Any], root: Path) -> dict[str, Any]:
    run_directory = root / str(candidate["run_id"])
    manifest_path = run_directory / "run_manifest.json"
    metrics_path = run_directory / "performance_metrics.json"
    outcome: dict[str, Any] = {"candidate_id": candidate["candidate_id"], "run_id": candidate["run_id"]}
    missing = [str(path.name) for path in (manifest_path, metrics_path) if not path.exists()]
    if missing:
        return {**outcome, "status": "missing_artifacts", "missing_artifacts": missing}
    try:
        manifest = _load_json_object(manifest_path, "run manifest")
        metrics = _load_json_object(metrics_path, "performance metrics")
    except ValueError as error:
        return {**outcome, "status": "invalid_artifacts", "error": str(error)}
    mismatches = _manifest_mismatches(manifest, candidate, ledger)
    metric = metrics.get(str(ledger["selection_metric"]))
    if not isinstance(metric, (float, int)) or not np.isfinite(float(metric)):
        mismatches.append(f"performance_metrics.{ledger['selection_metric']} must be finite")
    if mismatches:
        return {**outcome, "status": "manifest_or_metric_mismatch", "mismatches": mismatches}
    return {**outcome, "status": "complete", "metric_value": float(metric)}


def _manifest_mismatches(
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if manifest.get("run_id") != candidate["run_id"]:
        mismatches.append("run_id")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        return [*mismatches, "extra"]
    if extra.get("study_scope") != ledger["study_scope"]:
        mismatches.append("extra.study_scope")
    contract = extra.get("data_contract")
    if not isinstance(contract, Mapping) or contract.get("contract_id") != ledger["data_contract_id"]:
        mismatches.append("extra.data_contract.contract_id")
    walk_forward = extra.get("walk_forward")
    if not isinstance(walk_forward, Mapping):
        mismatches.append("extra.walk_forward")
    else:
        for field, expected in ledger["validation_window"].items():
            observed = walk_forward.get(field)
            if expected is None:
                # A declared open end must be an observed open end; a run that
                # silently bounded itself is not the run that was declared.
                if observed is not None:
                    mismatches.append(f"extra.walk_forward.{field}")
                continue
            if observed is None:
                mismatches.append(f"extra.walk_forward.{field}")
                continue
            try:
                observed_date = pd.Timestamp(observed).normalize()
            except (TypeError, ValueError):
                mismatches.append(f"extra.walk_forward.{field}")
                continue
            if observed_date != expected:
                mismatches.append(f"extra.walk_forward.{field}")
    for path, expected in candidate["expected_manifest_values"].items():
        if _nested_value(manifest, path) != expected:
            mismatches.append(path)
    return mismatches


def _nested_value(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload
