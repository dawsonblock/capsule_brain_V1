"""Row count integrity checks for v0.4.5 evidence packages.

For every JSONL artifact, count actual rows and compare to declared counts.

Required checks:

- actual counterfactual rows == declared ``n_outcomes``
- actual experience rows == declared ``n_experiences``
- actual safety rows == declared ``n_safety_tasks``

Also reported:

- unique task count
- unique action count
- rows by split
- rows by action
- rows by status
- rows by verifier
- rows by model revision
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _count_real_rows(records: list[dict[str, Any]]) -> int:
    """Count rows that look like actual task-level records.

    A summary row carrying only ``n_outcomes`` / ``n_experiences`` and a
    ``note`` is treated as a structural placeholder, not a real row.
    """
    real = [
        r for r in records
        if isinstance(r, dict) and ("task_id" in r or "outcome" in r or "action" in r)
    ]
    return len(real) if real else len(records)


def _safe_get(record: dict[str, Any], *keys: str) -> Any:
    """Safishly traverse nested keys in a record."""
    current: Any = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------


def check_row_counts(evidence_dir: str | Path) -> dict[str, Any]:
    """Count actual JSONL rows and compare to declared counts.

    Parameters
    ----------
    evidence_dir:
        Path to the immutable evidence package directory.

    Returns
    -------
    dict
        A result dict with keys:

        - ``status``: ``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"``
        - ``counterfactual_rows``: ``{actual, declared, match}``
        - ``experience_rows``: ``{actual, declared, match}``
        - ``safety_rows``: ``{actual, declared, match}``
        - ``unique_task_count``: int
        - ``unique_action_count``: int
        - ``rows_by_split``: dict
        - ``rows_by_action``: dict
        - ``rows_by_status``: dict
        - ``rows_by_verifier``: dict
        - ``rows_by_model_revision``: dict
        - ``errors``: list
    """
    edir = Path(evidence_dir)
    errors: list[str] = []

    if not edir.exists() or not edir.is_dir():
        return {
            "status": "BLOCKED",
            "counterfactual_rows": {"actual": 0, "declared": None, "match": False},
            "experience_rows": {"actual": 0, "declared": None, "match": False},
            "safety_rows": {"actual": 0, "declared": None, "match": False},
            "unique_task_count": 0,
            "unique_action_count": 0,
            "rows_by_split": {},
            "rows_by_action": {},
            "rows_by_status": {},
            "rows_by_verifier": {},
            "rows_by_model_revision": {},
            "errors": [f"Evidence directory does not exist: {edir}"],
        }

    # --- Load JSONL artifacts ---------------------------------------------
    cf_records = _load_jsonl(edir / "counterfactual_outcomes.jsonl")
    exp_records = _load_jsonl(edir / "executive_experiences.jsonl")

    # safety_results may be JSONL or JSON; handle both.
    safety_path = edir / "safety_results.jsonl"
    if not safety_path.exists():
        safety_path = edir / "safety_results.json"
    safety_records: list[dict[str, Any]] = []
    if safety_path.suffix == ".jsonl":
        safety_records = _load_jsonl(safety_path)
    else:
        safety_json = _load_json(safety_path)
        if isinstance(safety_json, list):
            safety_records = safety_json
        elif isinstance(safety_json, dict):
            # A single aggregate safety record is treated as one row.
            safety_records = [safety_json]

    # --- Load manifests for declared counts -------------------------------
    benchmark_manifest = _load_json(edir / "benchmark_manifest.json") or {}
    dataset_manifest = _load_json(edir / "dataset_manifest.json") or {}
    coverage_results = _load_json(edir / "coverage_results.json") or {}
    provider_manifest = _load_json(edir / "provider_manifest.json") or {}
    split_manifest = _load_json(edir / "split_manifest.json") or {}

    # --- Counterfactual rows ----------------------------------------------
    declared_n_outcomes: Any = None
    if cf_records and isinstance(cf_records[0], dict):
        declared_n_outcomes = cf_records[0].get("n_outcomes")
    if declared_n_outcomes is None:
        declared_n_outcomes = dataset_manifest.get("n_counterfactuals")
    if declared_n_outcomes is None:
        declared_n_outcomes = coverage_results.get("n_counterfactuals")
    actual_cf_rows = _count_real_rows(cf_records)
    cf_match = (
        isinstance(declared_n_outcomes, int)
        and actual_cf_rows == declared_n_outcomes
    )
    if not cf_match and isinstance(declared_n_outcomes, int):
        errors.append(
            f"Counterfactual row count mismatch: actual={actual_cf_rows} "
            f"declared={declared_n_outcomes}"
        )

    # --- Experience rows --------------------------------------------------
    declared_n_experiences: Any = None
    if exp_records and isinstance(exp_records[0], dict):
        declared_n_experiences = exp_records[0].get("n_experiences")
    if declared_n_experiences is None:
        declared_n_experiences = benchmark_manifest.get("n_experience")
    actual_exp_rows = _count_real_rows(exp_records)
    exp_match = (
        isinstance(declared_n_experiences, int)
        and actual_exp_rows == declared_n_experiences
    )
    if not exp_match and isinstance(declared_n_experiences, int):
        errors.append(
            f"Experience row count mismatch: actual={actual_exp_rows} "
            f"declared={declared_n_experiences}"
        )

    # --- Safety rows ------------------------------------------------------
    declared_n_safety: Any = None
    if safety_records and isinstance(safety_records[0], dict):
        declared_n_safety = safety_records[0].get("n_safety_tasks")
    if declared_n_safety is None:
        declared_n_safety = benchmark_manifest.get("n_safety")
    actual_safety_rows = _count_real_rows(safety_records)
    safety_match = (
        isinstance(declared_n_safety, int)
        and actual_safety_rows == declared_n_safety
    )
    if not safety_match and isinstance(declared_n_safety, int):
        errors.append(
            f"Safety row count mismatch: actual={actual_safety_rows} "
            f"declared={declared_n_safety}"
        )

    # --- Aggregations across all JSONL records ---------------------------
    all_records: list[dict[str, Any]] = []
    all_records.extend(cf_records)
    all_records.extend(exp_records)
    all_records.extend(safety_records)

    task_ids: set[Any] = set()
    actions: set[Any] = set()
    rows_by_split: Counter[str] = Counter()
    rows_by_action: Counter[str] = Counter()
    rows_by_status: Counter[str] = Counter()
    rows_by_verifier: Counter[str] = Counter()
    rows_by_model_revision: Counter[str] = Counter()

    model_revision = provider_manifest.get("model_revision") if isinstance(provider_manifest, dict) else None

    for record in all_records:
        if not isinstance(record, dict):
            continue
        task_id = record.get("task_id")
        if task_id is not None:
            task_ids.add(task_id)
        action = record.get("action")
        if action is not None:
            actions.add(action)
            rows_by_action[str(action)] += 1
        split = record.get("split")
        if split is not None:
            rows_by_split[str(split)] += 1
        status = record.get("action_status") or record.get("status")
        if status is not None:
            rows_by_status[str(status)] += 1
        verifier = record.get("verifier") or record.get("verifier_version")
        if verifier is not None:
            rows_by_verifier[str(verifier)] += 1
        rev = record.get("model_revision") or model_revision
        if rev is not None:
            rows_by_model_revision[str(rev)] += 1

    # Also fold in task_ids declared in the split manifest.
    splits = split_manifest.get("splits") if isinstance(split_manifest, dict) else None
    if isinstance(splits, dict):
        for split_info in splits.values():
            if isinstance(split_info, dict) and isinstance(split_info.get("task_ids"), list):
                task_ids.update(split_info["task_ids"])

    # --- Determine overall status -----------------------------------------
    required_matches = [cf_match, exp_match, safety_match]
    all_match = all(required_matches)
    status = "PASS" if (all_match and not errors) else "FAIL"

    return {
        "status": status,
        "counterfactual_rows": {
            "actual": actual_cf_rows,
            "declared": declared_n_outcomes,
            "match": cf_match,
        },
        "experience_rows": {
            "actual": actual_exp_rows,
            "declared": declared_n_experiences,
            "match": exp_match,
        },
        "safety_rows": {
            "actual": actual_safety_rows,
            "declared": declared_n_safety,
            "match": safety_match,
        },
        "unique_task_count": len(task_ids),
        "unique_action_count": len(actions),
        "rows_by_split": dict(rows_by_split),
        "rows_by_action": dict(rows_by_action),
        "rows_by_status": dict(rows_by_status),
        "rows_by_verifier": dict(rows_by_verifier),
        "rows_by_model_revision": dict(rows_by_model_revision),
        "errors": errors,
    }
