"""Data loading for v0.4.4 evidence-complete analysis.

Loads evidence package artifacts into typed Python objects.  This module is
the single entry point for reading an immutable evidence directory and
returning structured, validated data.

Key design decisions (defect #12 fix):
- Missing ``utility`` on an executed outcome raises ``MissingUtilityError``.
- A measured zero (``0.0``) remains a valid utility value.
- Not-executed actions use ``None`` (not ``0.0``) for their utility.
- We never use ``float(record.get("utility", 0.0))``; instead we perform
  explicit presence/null validation before coercion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.schemas import (
    InvalidOutcomeError,
    MissingUtilityError,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    split: str
    family: str
    archetype: str
    eligible_actions: list[str]


@dataclass(frozen=True)
class PolicyInfo:
    policy_id: str
    policy_type: str
    learner: str
    training_split: str
    seed: int


@dataclass(frozen=True)
class PolicyResults:
    policy_id: str
    mean_utility: float | None
    n_tasks: int
    verified_success_rate: float | None


@dataclass(frozen=True)
class CounterfactualOutcome:
    task_id: str
    action: str
    utility: float | None
    outcome_status: str  # "EXECUTED" | "NOT_EXECUTED" | "TIMEOUT" | "ERROR"
    verified_success: bool | None


@dataclass(frozen=True)
class LoadedEvidence:
    evidence_dir: Path
    evidence_manifest: dict
    benchmark_manifest: dict
    split_manifest: dict
    provider_manifest: dict
    dataset_manifest: dict
    candidate_policy: dict
    sham_policy: dict
    baseline_results: dict
    candidate_results: dict
    sham_results: dict
    oracle_results: dict
    gate_a_results: dict
    ood_results: dict
    safety_results: dict
    coverage_results: dict
    runtime_completion_diagnostics: dict
    source_provenance: dict
    counterfactual_outcomes: list[dict]
    executive_experiences: list[dict]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load a JSON file as a dict, raising a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required evidence artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(data).__name__}"
        )
    return data


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning the list of record dicts.

    The evidence-package JSONL files begin with a header line containing
    run-level metadata (``run_id``, ``note``, ``n_outcomes``/``n_experiences``).
    That header is *not* a per-record entry; it is identified by the absence of
    a ``task_id`` key and skipped.  Every subsequent non-blank line must be a
    JSON object and is appended to the returned list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required evidence artifact not found: {path}")
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object on line {lineno} of {path}, "
                    f"got {type(obj).__name__}"
                )
            # Skip the run-level header line (no task_id / action key).
            if "task_id" not in obj and "action" not in obj:
                continue
            records.append(obj)
    return records


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_tasks(benchmark_manifest: dict) -> dict[str, TaskInfo]:
    """Parse benchmark_manifest into task info objects.

    The benchmark manifest may carry per-task entries under a ``tasks`` key
    (a list of task dicts or a mapping of task_id -> task dict).  When no
    per-task entries are present (as with the placeholder 7B evidence
    package), an empty mapping is returned.
    """
    tasks: dict[str, TaskInfo] = {}
    eligible_actions = list(benchmark_manifest.get("eligible_actions", []))

    raw_tasks = benchmark_manifest.get("tasks")
    if raw_tasks is None:
        return tasks

    if isinstance(raw_tasks, dict):
        iterable = raw_tasks.values()
    elif isinstance(raw_tasks, list):
        iterable = raw_tasks
    else:
        raise ValueError(
            "benchmark_manifest['tasks'] must be a list or mapping, "
            f"got {type(raw_tasks).__name__}"
        )

    for entry in iterable:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Task entry must be a dict, got {type(entry).__name__}"
            )
        task_id = entry.get("task_id")
        if not task_id:
            raise ValueError("Task entry missing 'task_id'")
        tasks[task_id] = TaskInfo(
            task_id=str(task_id),
            split=str(entry.get("split", "")),
            family=str(entry.get("family", "")),
            archetype=str(entry.get("archetype", "")),
            eligible_actions=list(entry.get("eligible_actions", eligible_actions)),
        )
    return tasks


def parse_split_task_ids(split_manifest: dict) -> dict[str, list[str]]:
    """Parse split_manifest into per-split task ID lists."""
    splits_raw = split_manifest.get("splits", {})
    if not isinstance(splits_raw, dict):
        raise ValueError(
            "split_manifest['splits'] must be a mapping, "
            f"got {type(splits_raw).__name__}"
        )
    result: dict[str, list[str]] = {}
    for split_name, split_info in splits_raw.items():
        if not isinstance(split_info, dict):
            raise ValueError(
                f"Split '{split_name}' must be a dict, "
                f"got {type(split_info).__name__}"
            )
        task_ids = split_info.get("task_ids", [])
        if not isinstance(task_ids, list):
            raise ValueError(
                f"Split '{split_name}' task_ids must be a list, "
                f"got {type(task_ids).__name__}"
            )
        result[split_name] = [str(tid) for tid in task_ids]
    return result


def parse_policy(record: dict) -> PolicyInfo:
    """Parse a policy manifest dict into a PolicyInfo object."""
    return PolicyInfo(
        policy_id=str(record.get("policy_id", "")),
        policy_type=str(record.get("policy_type", "")),
        learner=str(record.get("learner", "")),
        training_split=str(record.get("training_split", "")),
        seed=int(record.get("seed", 0)),
    )


def parse_policy_results(record: dict) -> PolicyResults:
    """Parse a results dict into a PolicyResults object.

    ``mean_utility`` and ``verified_success_rate`` may be absent (None) rather
    than defaulting to 0.0, so that "not measured" is distinguishable from
    "measured zero".
    """
    mean_utility = record.get("mean_utility")
    if mean_utility is not None:
        mean_utility = float(mean_utility)
    vsr = record.get("verified_success_rate")
    if vsr is not None:
        vsr = float(vsr)
    return PolicyResults(
        policy_id=str(record.get("policy_id", "")),
        mean_utility=mean_utility,
        n_tasks=int(record.get("n_tasks", 0)),
        verified_success_rate=vsr,
    )


def parse_outcome(record: dict) -> CounterfactualOutcome:
    """Parse a single counterfactual outcome record with strict utility validation.

    Defect #12 fix: never use ``float(record.get("utility", 0.0))``.  Instead:
    1. If ``utility`` key is absent entirely -> ``MissingUtilityError``.
    2. If ``outcome_status`` is ``EXECUTED`` and utility is ``None`` ->
       ``InvalidOutcomeError`` (an executed outcome must carry a measured
       value, even if that value is 0.0).
    3. Not-executed actions keep ``utility=None``.
    """
    if "utility" not in record:
        raise MissingUtilityError(
            f"Outcome record for task_id={record.get('task_id')!r} "
            f"action={record.get('action')!r} is missing the 'utility' key"
        )

    raw_utility = record["utility"]
    outcome_status = str(record.get("outcome_status", "EXECUTED"))

    if outcome_status == "EXECUTED" and raw_utility is None:
        raise InvalidOutcomeError(
            f"Outcome record for task_id={record.get('task_id')!r} "
            f"action={record.get('action')!r} has outcome_status=EXECUTED "
            f"but utility is None"
        )

    if raw_utility is None:
        utility: float | None = None
    else:
        utility = float(raw_utility)

    raw_verified = record.get("verified_success")
    if raw_verified is None:
        verified_success: bool | None = None
    else:
        verified_success = bool(raw_verified)

    return CounterfactualOutcome(
        task_id=str(record.get("task_id", "")),
        action=str(record.get("action", "")),
        utility=utility,
        outcome_status=outcome_status,
        verified_success=verified_success,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_evidence(evidence_dir: str | Path) -> LoadedEvidence:
    """Load all evidence package artifacts."""
    d = Path(evidence_dir)
    if not d.exists():
        raise FileNotFoundError(f"Evidence directory not found: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Evidence path is not a directory: {d}")

    return LoadedEvidence(
        evidence_dir=d,
        evidence_manifest=_load_json(d / "EVIDENCE_MANIFEST.json"),
        benchmark_manifest=_load_json(d / "benchmark_manifest.json"),
        split_manifest=_load_json(d / "split_manifest.json"),
        provider_manifest=_load_json(d / "provider_manifest.json"),
        dataset_manifest=_load_json(d / "dataset_manifest.json"),
        candidate_policy=_load_json(d / "candidate_policy.json"),
        sham_policy=_load_json(d / "sham_policy.json"),
        baseline_results=_load_json(d / "baseline_results.json"),
        candidate_results=_load_json(d / "candidate_results.json"),
        sham_results=_load_json(d / "sham_results.json"),
        oracle_results=_load_json(d / "oracle_results.json"),
        gate_a_results=_load_json(d / "gate_a_results.json"),
        ood_results=_load_json(d / "ood_results.json"),
        safety_results=_load_json(d / "safety_results.json"),
        coverage_results=_load_json(d / "coverage_results.json"),
        runtime_completion_diagnostics=_load_json(
            d / "runtime_completion_diagnostics.json"
        ),
        source_provenance=_load_json(d / "source_provenance.json"),
        counterfactual_outcomes=_load_jsonl(d / "counterfactual_outcomes.jsonl"),
        executive_experiences=_load_jsonl(d / "executive_experiences.jsonl"),
    )
