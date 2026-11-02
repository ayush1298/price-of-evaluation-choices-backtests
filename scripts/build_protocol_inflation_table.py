"""Measure how much each evaluation shortcut inflates a reported result.

Every model candidate in this project failed its held-out gate.  That makes the
*protocol* the measurable object: given the same manifest-backed runs, how much
higher does a reported number look when read under a permissive convention
instead of the strict one?

This script answers that from run artefacts only.  It never refits, reprices, or
selects a strategy: it reads `performance_metrics.json` and `run_manifest.json`
from directories that already exist, applies a pre-declared reading rule to
each side of a matched pair, and records the SHA-256 of every input it used.

The spec is versioned so the set of rows cannot be quietly grown after seeing
which rows look most impressive.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.reproducibility import sha256_file

AGGREGATORS = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "max": max,
    "min": min,
}


def _project_path(value: str | Path) -> Path:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Expected valid JSON: {path}") from error


def _load_side(
    side: Mapping[str, Any],
    *,
    results_root: Path,
    metric: str,
) -> dict[str, Any]:
    """Read one side of a matched pair and reduce it with its declared rule."""
    run_ids = side.get("runs")
    if not isinstance(run_ids, list) or not run_ids:
        raise ValueError("Each side of a row must name a non-empty 'runs' list")
    rule = str(side.get("rule", "mean"))
    if rule not in AGGREGATORS:
        raise ValueError(f"Unknown reading rule {rule!r}; expected one of {sorted(AGGREGATORS)}")

    values: list[float] = []
    observations: set[int] = set()
    inputs: dict[str, dict[str, str]] = {}
    for run_id in run_ids:
        directory = results_root / str(run_id)
        metrics_path = directory / "performance_metrics.json"
        manifest_path = directory / "run_manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Run {run_id!r} lacks performance_metrics.json/run_manifest.json; "
                "the table refuses to silently drop a named run"
            )
        metrics = _read_json(metrics_path)
        if metric not in metrics:
            raise ValueError(f"Run {run_id!r} has no {metric!r} metric")
        values.append(float(metrics[metric]))
        observations.add(int(metrics["n_observations"]))
        inputs[str(run_id)] = {
            "run_manifest_sha256": sha256_file(manifest_path),
            "performance_metrics_sha256": sha256_file(metrics_path),
            metric: float(metrics[metric]),
        }

    # A shortcut row is only meaningful if the runs it reduces share a period.
    if len(observations) != 1:
        raise ValueError(
            f"Runs {run_ids} span different evaluation lengths {sorted(observations)}; "
            "a reading rule may only reduce runs over one matched period"
        )

    return {
        "rule": rule,
        "value": float(AGGREGATORS[rule](values)),
        "n_runs": len(values),
        "n_observations": observations.pop(),
        "runs": inputs,
    }


def _build_row(row: Mapping[str, Any], *, results_root: Path, metric: str) -> dict[str, Any]:
    reported = _load_side(row["reported"], results_root=results_root, metric=metric)
    honest = _load_side(row["honest"], results_root=results_root, metric=metric)
    payload: dict[str, Any] = {
        "row_id": row["row_id"],
        "shortcut": row["shortcut"],
        "panel": row["panel"],
        "strategy": row["strategy"],
        "reported": reported,
        "honest": honest,
        "inflation": reported["value"] - honest["value"],
    }
    reference = row.get("reference")
    if isinstance(reference, Mapping):
        resolved = _load_side(reference, results_root=results_root, metric=metric)
        resolved["label"] = reference.get("label")
        payload["reference"] = resolved
    return payload


def _build_benchmark_choice_row(
    row: Mapping[str, Any], *, results_root: Path, metric: str
) -> dict[str, Any]:
    """Measure what fixing one favourite baseline buys, versus a baseline ladder.

    Unlike the reading-rule rows this compares two *differences*, because the
    quantity a paper reports is a candidate's edge over a benchmark, and that
    edge moves when the benchmark is chosen rather than enumerated.
    """
    def _sharpe(run_id: str) -> float:
        metrics = _read_json(results_root / run_id / "performance_metrics.json")
        return float(metrics[metric])

    candidate = str(row["candidate_run"])
    convenient = str(row["convenient_benchmark_run"])
    pool = [str(run_id) for run_id in row["baseline_pool"]]
    if convenient not in pool:
        raise ValueError("The convenient benchmark must itself be one of the declared baselines")

    candidate_value = _sharpe(candidate)
    pool_values = {run_id: _sharpe(run_id) for run_id in pool}
    strongest = max(pool_values, key=pool_values.get)

    reported_edge = candidate_value - pool_values[convenient]
    honest_edge = candidate_value - pool_values[strongest]
    return {
        "row_id": row["row_id"],
        "shortcut": row["shortcut"],
        "panel": row["panel"],
        "strategy": row["strategy"],
        "candidate_run": candidate,
        "candidate_value": candidate_value,
        "convenient_benchmark": {"run_id": convenient, metric: pool_values[convenient]},
        "strongest_baseline": {"run_id": strongest, metric: pool_values[strongest]},
        "baseline_pool": pool_values,
        "reported_edge": reported_edge,
        "honest_edge": honest_edge,
        "inflation": reported_edge - honest_edge,
        "conclusion_sign_flips": (reported_edge > 0) != (honest_edge > 0),
    }


def _build_period_mismatch_row(
    row: Mapping[str, Any], *, results_root: Path, metric: str
) -> dict[str, Any]:
    """Measure what comparing against a differently-dated benchmark buys.

    A model with a long warm-up starts later than its baselines.  Reading both
    numbers off the same table compares different market periods, which is not
    a modelling error the reader can see: every individual run is correct.
    The reported edge here uses the benchmark as printed; the honest edge uses
    the same benchmark re-run over the candidate's own window.
    """
    def _metrics(run_id: str) -> dict[str, Any]:
        return _read_json(results_root / run_id / "performance_metrics.json")

    candidate = _metrics(str(row["candidate_run"]))
    mismatched = _metrics(str(row["mismatched_benchmark_run"]))
    matched = _metrics(str(row["matched_benchmark_run"]))
    if int(mismatched["n_observations"]) == int(matched["n_observations"]):
        raise ValueError(
            "The mismatched and matched benchmarks cover the same period; "
            "this row would not demonstrate anything"
        )
    reported_edge = float(candidate[metric]) - float(mismatched[metric])
    honest_edge = float(candidate[metric]) - float(matched[metric])
    return {
        "row_id": row["row_id"],
        "shortcut": row["shortcut"],
        "panel": row["panel"],
        "strategy": row["strategy"],
        "candidate": {"run_id": row["candidate_run"], metric: float(candidate[metric]),
                      "n_observations": int(candidate["n_observations"])},
        "mismatched_benchmark": {"run_id": row["mismatched_benchmark_run"], metric: float(mismatched[metric]),
                                 "n_observations": int(mismatched["n_observations"])},
        "matched_benchmark": {"run_id": row["matched_benchmark_run"], metric: float(matched[metric]),
                              "n_observations": int(matched["n_observations"])},
        "reported_edge": reported_edge,
        "honest_edge": honest_edge,
        "inflation": reported_edge - honest_edge,
        "conclusion_sign_flips": (reported_edge > 0) != (honest_edge > 0),
    }


def _build_inference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    check_path = _project_path(str(row["reality_check"]))
    check = _read_json(check_path)
    result = check["result"]
    return {
        "row_id": row["row_id"],
        "shortcut": row["shortcut"],
        "panel": row["panel"],
        "strategy": row["strategy"],
        "reported": {
            "label": row["reported_label"],
            "p_value": float(row["reported_p_value"]),
            "source": row["reported_p_value_source"],
        },
        "honest": {
            "label": row["honest_label"],
            "p_value": float(result["reality_check_p_value"]),
            "candidates": int(result["candidates"]),
            "n_observations": int(result["observations"]),
            "best_candidate": result["best_candidate"],
            "reality_check_sha256": sha256_file(check_path),
        },
        "p_value_understatement": float(result["reality_check_p_value"]) - float(row["reported_p_value"]),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    metric = payload["primary_metric"]
    lines = [
        "| Evaluation shortcut | Panel / strategy | Reported | Honest | Inflation |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        reported = f"{row['reported']['value']:.3f} ({row['reported']['rule']})"
        honest = f"{row['honest']['value']:.3f} ({row['honest']['rule']})"
        lines.append(
            f"| {row['shortcut']} | {row['panel']}: {row['strategy']} | "
            f"{reported} | {honest} | **{row['inflation']:+.3f}** |"
        )
    for row in payload.get("benchmark_choice_rows", []):
        flip = " **sign flips**" if row["conclusion_sign_flips"] else ""
        lines.append(
            f"| {row['shortcut']} | {row['panel']}: {row['strategy']} | "
            f"{row['reported_edge']:+.3f} edge | {row['honest_edge']:+.3f} edge | "
            f"**{row['inflation']:+.3f}**{flip} |"
        )
    for row in payload.get("period_mismatch_rows", []):
        lines.append(
            f"| {row['shortcut']} | {row['panel']}: {row['strategy']} | "
            f"{row['reported_edge']:+.3f} edge | {row['honest_edge']:+.3f} edge | "
            f"**{row['inflation']:+.3f}** |"
        )
    for row in payload["inference_rows"]:
        lines.append(
            f"| {row['shortcut']} | {row['panel']}: {row['strategy']} | "
            f"p={row['reported']['p_value']:.3f} | p={row['honest']['p_value']:.3f} | "
            f"**{row['p_value_understatement']:+.3f}** |"
        )
    lines.append("")
    lines.append(f"All allocation rows report net {metric}. Inflation is reported minus honest.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/protocol_inflation_table_v1.json")
    parser.add_argument("--results-root", default="results/canonical")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec_path = _project_path(args.spec)
    output = _project_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite if intended")

    spec = _read_json(spec_path)
    if spec.get("kind") != "protocol-inflation-table-spec":
        raise ValueError(f"Not a protocol-inflation table spec: {spec_path}")
    results_root = _project_path(args.results_root)
    metric = str(spec.get("primary_metric", "sharpe"))

    payload = {
        "kind": "protocol-inflation-table",
        "spec": {"path": str(spec_path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(spec_path)},
        "spec_id": spec["spec_id"],
        "primary_metric": metric,
        "rows": [_build_row(row, results_root=results_root, metric=metric) for row in spec["rows"]],
        "benchmark_choice_rows": [
            _build_benchmark_choice_row(row, results_root=results_root, metric=metric)
            for row in spec.get("benchmark_choice_rows", [])
        ],
        "period_mismatch_rows": [
            _build_period_mismatch_row(row, results_root=results_root, metric=metric)
            for row in spec.get("period_mismatch_rows", [])
        ],
        "inference_rows": [_build_inference_row(row) for row in spec.get("inference_rows", [])],
        "warning": (
            "This table measures reporting conventions, not strategy quality. No row here is evidence "
            "that any method works; every candidate involved failed its held-out gate."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote protocol-inflation table to {output.relative_to(PROJECT_ROOT)}")

    if args.markdown_output:
        markdown_path = _project_path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(payload) + "\n", encoding="utf-8")
        print(f"Wrote markdown table to {markdown_path.relative_to(PROJECT_ROOT)}")
    print()
    print(_markdown(payload))


if __name__ == "__main__":
    main()
