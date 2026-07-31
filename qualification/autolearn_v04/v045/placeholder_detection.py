"""Placeholder detection for v0.4.5 evidence packages.

Reject evidence containing placeholder markers or structural placeholders.

This module does NOT rely only on marker strings.  It also detects structural
placeholders: manifest contradictions where a declared count, dimension, or
identity is present but the corresponding task-level data is absent, empty, or
inconsistent.

Two kinds of detection are performed:

1.  **Text markers** -- scan every evidence file for the configured
    ``PLACEHOLDER_MARKERS`` plus the additional marker ``"unknown"`` (which
    frequently appears in stale provenance fields such as
    ``original_commit_sha``).
2.  **Structural placeholders** -- manifest contradictions:

    - declared nonzero split count with zero task IDs
    - declared ``n_outcomes`` != actual JSONL row count
    - declared ``n_tasks`` != unique task ID count
    - declared policy dimension but missing weights
    - declared model digest but empty value
    - immutable run with unknown source identity
    - scientific result with no task-level outputs
    - policy evaluation with no task records
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.v045.config import PLACEHOLDER_MARKERS


# ---------------------------------------------------------------------------
# Marker vocabulary
# ---------------------------------------------------------------------------

# The config tuple covers the canonical placeholder phrases.  ``"unknown"`` is
# treated as an additional low-severity marker because it legitimately appears
# in some provenance fields but is a strong placeholder signal when found in
# commit hashes, run identities, or model digests.
_ADDITIONAL_MARKERS: tuple[str, ...] = ("unknown",)

_ALL_MARKERS: tuple[str, ...] = tuple(
    dict.fromkeys((*PLACEHOLDER_MARKERS, *_ADDITIONAL_MARKERS))
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file line by line, skipping blank lines."""
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def _read_text(path: Path) -> str:
    """Read a file as text, returning ``""`` on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _add_detection(
    detections: list[dict[str, Any]],
    file: str,
    reason: str,
    field: str,
    observed: Any,
    expected: Any,
    severity: str,
    status: str,
) -> None:
    """Append a structured detection record."""
    detections.append({
        "file": file,
        "reason": reason,
        "field": field,
        "observed": observed,
        "expected": expected,
        "severity": severity,
        "status": status,
    })


def _scan_text_markers(
    evidence_dir: Path,
    detections: list[dict[str, Any]],
) -> None:
    """Scan every evidence file for placeholder marker strings."""
    for path in sorted(evidence_dir.iterdir()):
        if not path.is_file():
            continue
        text = _read_text(path)
        if not text:
            continue
        lowered = text.lower()
        for marker in _ALL_MARKERS:
            if marker.lower() in lowered:
                # "unknown" is only a placeholder signal when it appears as a
                # value in provenance/identity fields, not as a generic word.
                # We still report it but at low severity.
                if marker.lower() == "unknown":
                    severity = "low"
                    status = "WARN"
                else:
                    severity = "high"
                    status = "FAIL"
                _add_detection(
                    detections,
                    file=path.name,
                    reason=f"placeholder marker '{marker}' found in file content",
                    field="content",
                    observed=marker,
                    expected="no placeholder markers",
                    severity=severity,
                    status=status,
                )


def _check_structural_placeholders(
    evidence_dir: Path,
    detections: list[dict[str, Any]],
) -> None:
    """Detect structural placeholders via manifest contradictions."""

    # --- EVIDENCE_MANIFEST.json -------------------------------------------
    manifest = _load_json(evidence_dir / "EVIDENCE_MANIFEST.json")
    if isinstance(manifest, dict):
        # declared model digest but empty value
        model_digest = manifest.get("model_digest")
        if model_digest is not None and model_digest == "":
            _add_detection(
                detections,
                file="EVIDENCE_MANIFEST.json",
                reason="declared model_digest but empty value",
                field="model_digest",
                observed=model_digest,
                expected="non-empty model digest",
                severity="medium",
                status="FAIL",
            )

        # immutable run with unknown source identity
        immutable = manifest.get("immutable")
        original_commit = manifest.get("original_commit_sha")
        if immutable is True and (
            original_commit in (None, "", "unknown")
            or (isinstance(original_commit, str) and "unknown" in original_commit.lower())
        ):
            _add_detection(
                detections,
                file="EVIDENCE_MANIFEST.json",
                reason="immutable run with unknown source identity",
                field="original_commit_sha",
                observed=original_commit,
                expected="known commit sha for immutable run",
                severity="high",
                status="FAIL",
            )

        # scientific result with no task-level outputs
        scientific_result = manifest.get("scientific_result")
        if scientific_result is not None and scientific_result != "":
            cf_records = _load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
            has_task_outputs = any(
                isinstance(r, dict) and ("task_id" in r or "task_ids" in r or "outcome" in r)
                for r in cf_records
            )
            if not has_task_outputs:
                _add_detection(
                    detections,
                    file="EVIDENCE_MANIFEST.json",
                    reason="scientific result declared with no task-level outputs",
                    field="scientific_result",
                    observed=scientific_result,
                    expected="task-level counterfactual outputs",
                    severity="high",
                    status="FAIL",
                )

    # --- provider_manifest.json: model digest empty ----------------------
    provider = _load_json(evidence_dir / "provider_manifest.json")
    if isinstance(provider, dict):
        p_digest = provider.get("model_digest")
        if p_digest is not None and p_digest == "":
            _add_detection(
                detections,
                file="provider_manifest.json",
                reason="declared model_digest but empty value",
                field="model_digest",
                observed=p_digest,
                expected="non-empty model digest",
                severity="medium",
                status="FAIL",
            )

    # --- split_manifest.json: nonzero count with zero task IDs -----------
    split_manifest = _load_json(evidence_dir / "split_manifest.json")
    if isinstance(split_manifest, dict):
        splits = split_manifest.get("splits")
        if isinstance(splits, dict):
            for split_name, split_info in splits.items():
                if not isinstance(split_info, dict):
                    continue
                count = split_info.get("count")
                task_ids = split_info.get("task_ids")
                if (
                    isinstance(count, int)
                    and count > 0
                    and isinstance(task_ids, list)
                    and len(task_ids) == 0
                ):
                    _add_detection(
                        detections,
                        file="split_manifest.json",
                        reason=(
                            f"declared nonzero count ({count}) for split "
                            f"'{split_name}' with zero task IDs"
                        ),
                        field=f"splits.{split_name}.task_ids",
                        observed=len(task_ids),
                        expected=count,
                        severity="high",
                        status="FAIL",
                    )

    # --- counterfactual_outcomes.jsonl: n_outcomes != actual rows --------
    cf_records = _load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
    dataset_manifest = _load_json(evidence_dir / "dataset_manifest.json")
    declared_n_outcomes: Any = None
    if cf_records:
        # n_outcomes may be a top-level field in the first record or in the
        # dataset/coverage manifests.
        declared_n_outcomes = cf_records[0].get("n_outcomes")
    if declared_n_outcomes is None:
        if isinstance(dataset_manifest, dict):
            declared_n_outcomes = dataset_manifest.get("n_counterfactuals")
    if declared_n_outcomes is None:
        coverage = _load_json(evidence_dir / "coverage_results.json")
        if isinstance(coverage, dict):
            declared_n_outcomes = coverage.get("n_counterfactuals")
    if isinstance(declared_n_outcomes, int):
        # Only count rows that look like actual outcome records (have a
        # task_id or outcome field).  A single summary row carrying only
        # n_outcomes is a structural placeholder.
        real_rows = [
            r for r in cf_records
            if isinstance(r, dict) and ("task_id" in r or "outcome" in r)
        ]
        actual_rows = len(real_rows) if real_rows else len(cf_records)
        if actual_rows != declared_n_outcomes:
            _add_detection(
                detections,
                file="counterfactual_outcomes.jsonl",
                reason="declared n_outcomes != actual JSONL row count",
                field="n_outcomes",
                observed=actual_rows,
                expected=declared_n_outcomes,
                severity="high",
                status="FAIL",
            )

    # --- benchmark_manifest / dataset_manifest: n_tasks != unique task IDs
    benchmark_manifest = _load_json(evidence_dir / "benchmark_manifest.json")
    declared_n_tasks: Any = None
    if isinstance(benchmark_manifest, dict):
        declared_n_tasks = benchmark_manifest.get("n_tasks")
    if declared_n_tasks is None and isinstance(dataset_manifest, dict):
        declared_n_tasks = dataset_manifest.get("n_tasks")
    if isinstance(declared_n_tasks, int):
        all_task_ids: set[Any] = set()
        for records in (cf_records, _load_jsonl(evidence_dir / "executive_experiences.jsonl")):
            for r in records:
                if isinstance(r, dict) and "task_id" in r:
                    all_task_ids.add(r["task_id"])
        # Also consider split_manifest task_ids.
        if isinstance(split_manifest, dict) and isinstance(split_manifest.get("splits"), dict):
            for split_info in split_manifest["splits"].values():
                if isinstance(split_info, dict) and isinstance(split_info.get("task_ids"), list):
                    all_task_ids.update(split_info["task_ids"])
        unique_task_count = len(all_task_ids)
        if unique_task_count != declared_n_tasks:
            _add_detection(
                detections,
                file="benchmark_manifest.json",
                reason="declared n_tasks != unique task ID count",
                field="n_tasks",
                observed=unique_task_count,
                expected=declared_n_tasks,
                severity="high",
                status="FAIL",
            )

    # --- policy dimension but missing weights ----------------------------
    for policy_file in ("candidate_policy.json", "sham_policy.json"):
        policy = _load_json(evidence_dir / policy_file)
        if isinstance(policy, dict):
            has_dimension = any(
                k in policy
                for k in ("n_training", "n_features", "n_dimensions", "policy_dimension")
            )
            has_weights = any(
                k in policy
                for k in ("weights", "coefficients", "params", "parameters", "theta")
            )
            if has_dimension and not has_weights:
                _add_detection(
                    detections,
                    file=policy_file,
                    reason="declared policy dimension but missing weights",
                    field="weights",
                    observed=None,
                    expected="non-empty weights/coefficients",
                    severity="medium",
                    status="FAIL",
                )

    # --- policy evaluation with no task records --------------------------
    for results_file in ("candidate_results.json", "sham_results.json", "baseline_results.json"):
        results = _load_json(evidence_dir / results_file)
        if isinstance(results, dict):
            n_tasks = results.get("n_tasks")
            if isinstance(n_tasks, int) and n_tasks > 0:
                # Look for any task-level records across the package.
                task_records = [
                    r for r in cf_records
                    if isinstance(r, dict) and "task_id" in r
                ]
                if not task_records:
                    _add_detection(
                        detections,
                        file=results_file,
                        reason="policy evaluation with no task records",
                        field="n_tasks",
                        observed=0,
                        expected=f"{n_tasks} task records",
                        severity="high",
                        status="FAIL",
                    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_placeholders(evidence_dir: str | Path) -> dict:
    """Detect placeholder markers and structural placeholders in evidence.

    Parameters
    ----------
    evidence_dir:
        Path to the immutable evidence package directory.

    Returns
    -------
    dict
        A result dict with keys:

        - ``status``: ``"PASS"`` | ``"FAIL"``
        - ``detections``: list of detection records, each containing
          ``file``, ``reason``, ``field``, ``observed``, ``expected``,
          ``severity`` (``"high"`` | ``"medium"`` | ``"low"``) and
          ``status`` (``"FAIL"`` | ``"WARN"``).
        - ``n_detections``: number of detections.
    """
    edir = Path(evidence_dir)
    detections: list[dict[str, Any]] = []

    if not edir.exists() or not edir.is_dir():
        _add_detection(
            detections,
            file=str(edir),
            reason="evidence directory does not exist",
            field="evidence_dir",
            observed=str(edir),
            expected="existing evidence directory",
            severity="high",
            status="FAIL",
        )
        return {
            "status": "FAIL",
            "detections": detections,
            "n_detections": len(detections),
        }

    # 1. Text marker scan across all files.
    _scan_text_markers(edir, detections)

    # 2. Structural placeholder detection via manifest contradictions.
    _check_structural_placeholders(edir, detections)

    # Overall status: FAIL if any detection has status "FAIL", otherwise PASS
    # (WARN-only detections do not fail the package on their own).
    has_fail = any(d["status"] == "FAIL" for d in detections)
    status = "FAIL" if has_fail else "PASS"

    return {
        "status": status,
        "detections": detections,
        "n_detections": len(detections),
    }
