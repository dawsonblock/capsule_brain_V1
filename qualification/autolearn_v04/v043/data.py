"""Data loading for v0.4.3 repair analysis.

Loads the frozen 7B run artifacts and provides structured access to:
- task definitions
- counterfactual outcomes
- evaluation results (the actual executed policy metrics)
- candidate/sham/baseline/oracle policies
- training/test examples with features

Key difference from v0.4.2: this module reads the *actual evaluation artifacts*
(evaluate_test_*.json) for policy mean utilities, rather than recomputing
them from stored weights.  This fixes the root cause of the -6.11
recovered-headroom defect.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    split: str
    family: str
    archetype: str
    allowed_actions: list[str]
    verifier_spec: dict[str, Any]
    prompt: str
    generator_family: str
    template_family: str
    capability_family: str
    entity_family: str
    risk_class: str
    crossover_type: str
    group_id: str
    generator_seed: int
    setup_spec: dict[str, Any]
    expected_output_digest: str


@dataclass(frozen=True)
class Outcome:
    task_id: str
    action_id: str
    availability: str  # "executed" | "unavailable" | "skipped"
    verification: str  # "success" | "failure" | "error" | "unknown"
    utility: float
    execution_metadata: dict[str, Any]
    reward_components: dict[str, Any]


@dataclass(frozen=True)
class PolicyEvaluation:
    """The actual evaluation results from evaluate_test_*.json."""
    policy_id: str
    split: str
    mean_utility: float
    success_rate: float
    n_complete_tasks: int
    n_manifest_tasks: int
    n_missing_tasks: int
    per_task: dict[str, dict[str, Any]]  # task_id -> {action, utility, success, ...}
    metrics: dict[str, Any]
    raw_artifact: dict[str, Any]


@dataclass
class LoadedRun:
    """Structured access to a frozen 7B run's artifacts."""
    artifacts_dir: Path
    benchmark_manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    counterfactual_outcomes: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    outcomes: list[Outcome] = field(default_factory=list)
    outcomes_by_task: dict[str, list[Outcome]] = field(default_factory=dict)
    outcomes_by_task_action: dict[tuple[str, str], Outcome] = field(default_factory=dict)

    # Actual evaluation artifacts (ground truth for policy metrics).
    eval_candidate: PolicyEvaluation | None = None
    eval_sham: PolicyEvaluation | None = None
    eval_baseline: PolicyEvaluation | None = None
    eval_oracle: PolicyEvaluation | None = None

    # Policies.
    candidate_policy: dict[str, Any] = field(default_factory=dict)
    sham_policy: dict[str, Any] = field(default_factory=dict)
    candidate_training: dict[str, Any] = field(default_factory=dict)
    sham_training: dict[str, Any] = field(default_factory=dict)

    # Gate A result.
    gate_a_result: dict[str, Any] = field(default_factory=dict)

    # Qualification report.
    qualification_report: dict[str, Any] = field(default_factory=dict)

    # Dataset examples.
    train_examples: list[dict[str, Any]] = field(default_factory=list)
    validation_examples: list[dict[str, Any]] = field(default_factory=list)
    test_examples: list[dict[str, Any]] = field(default_factory=list)
    ood_examples: list[dict[str, Any]] = field(default_factory=list)
    safety_examples: list[dict[str, Any]] = field(default_factory=list)

    # Manifests.
    dataset_manifest: dict[str, Any] = field(default_factory=dict)

    # SHA-256 digests.
    benchmark_manifest_sha256: str = ""
    split_manifest_sha256: str = ""
    counterfactual_results_sha256: str = ""
    candidate_policy_sha256: str = ""
    sham_policy_sha256: str = ""
    gate_a_result_sha256: str = ""
    qualification_report_sha256: str = ""

    # Convenience accessors.
    all_actions: list[str] = field(default_factory=list)
    all_families: list[str] = field(default_factory=list)
    all_archetypes: list[str] = field(default_factory=list)
    all_splits: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def test_task_ids(self) -> list[str]:
        return sorted(tid for tid, t in self.tasks.items() if t.split == "test")

    @property
    def validation_task_ids(self) -> list[str]:
        return sorted(tid for tid, t in self.tasks.items() if t.split == "validation")

    @property
    def train_task_ids(self) -> list[str]:
        return sorted(tid for tid, t in self.tasks.items() if t.split == "experience")

    def get_utility_for_action(self, task_id: str, action: str) -> float | None:
        o = self.outcomes_by_task_action.get((task_id, action))
        return o.utility if o and o.availability == "executed" else None

    def get_oracle_utility(self, task_id: str) -> float:
        """Max utility among executed outcomes for a task."""
        best = None
        for o in self.outcomes_by_task.get(task_id, []):
            if o.availability == "executed":
                if best is None or o.utility > best:
                    best = o.utility
        return best if best is not None else 0.0

    def get_oracle_action(self, task_id: str) -> str | None:
        """Action with highest measured utility for a task."""
        best_action = None
        best_u = None
        for o in self.outcomes_by_task.get(task_id, []):
            if o.availability == "executed":
                if best_u is None or o.utility > best_u:
                    best_u = o.utility
                    best_action = o.action_id
        return best_action

    def get_utility_matrix(self, task_ids: list[str]) -> tuple[list[str], list[str], np.ndarray]:
        """Return (task_ids, actions, U) where U[i,j] is utility for task i, action j.

        Unavailable outcomes are NaN.
        """
        actions = sorted(self.all_actions)
        action_idx = {a: i for i, a in enumerate(actions)}
        U = np.full((len(task_ids), len(actions)), np.nan)
        for i, tid in enumerate(task_ids):
            for o in self.outcomes_by_task.get(tid, []):
                if o.availability == "executed" and o.action_id in action_idx:
                    U[i, action_idx[o.action_id]] = o.utility
        return task_ids, actions, U


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(artifacts_dir: str | Path) -> LoadedRun:
    """Load a frozen run from its artifacts directory."""
    d = Path(artifacts_dir)
    if not d.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {d}")

    run = LoadedRun(
        artifacts_dir=d,
        benchmark_manifest=_load_json(d / "benchmark_manifest.json"),
        split_manifest=_load_json(d / "split_manifest.json"),
    )

    # SHA-256 digests of key artifacts.
    run.benchmark_manifest_sha256 = _sha256_file(d / "benchmark_manifest.json")
    run.split_manifest_sha256 = _sha256_file(d / "split_manifest.json")
    run.counterfactual_results_sha256 = _sha256_file(d / "counterfactual_outcomes.json")

    # Load counterfactual outcomes.
    raw_outcomes = _load_json(d / "counterfactual_outcomes.json")
    run.counterfactual_outcomes = raw_outcomes
    for ro in raw_outcomes:
        tid = ro.get("task_id", "")
        aid = ro.get("action_id", "")
        avail = ro.get("availability", "executed")
        verif = ro.get("verification", "unknown")
        util = float(ro.get("utility", 0.0))
        em = ro.get("execution_metadata", {})
        rc = ro.get("reward_components", {})
        o = Outcome(
            task_id=tid, action_id=aid, availability=avail,
            verification=verif, utility=util,
            execution_metadata=em, reward_components=rc,
        )
        run.outcomes.append(o)
        run.outcomes_by_task.setdefault(tid, []).append(o)
        run.outcomes_by_task_action[(tid, aid)] = o

    # Load tasks from benchmark manifest.
    for t in run.benchmark_manifest.get("tasks", []):
        tid = t.get("task_id", "")
        if not tid:
            continue
        ti = TaskInfo(
            task_id=tid,
            split=t.get("split", ""),
            family=t.get("family", ""),
            archetype=t.get("archetype", ""),
            allowed_actions=t.get("allowed_actions", []),
            verifier_spec=t.get("verifier_spec", {}),
            prompt=t.get("prompt", ""),
            generator_family=t.get("generator_family", ""),
            template_family=t.get("template_family", ""),
            capability_family=t.get("capability_family", ""),
            entity_family=t.get("entity_family", ""),
            risk_class=t.get("risk_class", "benign"),
            crossover_type=t.get("crossover_type", "none"),
            group_id=t.get("group_id", ""),
            generator_seed=t.get("generator_seed", 0),
            setup_spec=t.get("setup_spec", {}),
            expected_output_digest=t.get("expected_output_digest", ""),
        )
        run.tasks[tid] = ti

    # Load evaluation artifacts (the ground truth for policy metrics).
    run.eval_candidate = _load_policy_eval(d, "evaluate_test_candidate.json", "candidate")
    run.eval_sham = _load_policy_eval(d, "evaluate_test_sham.json", "sham")
    run.eval_baseline = _load_policy_eval(d, "evaluate_test_baseline.json", "baseline")
    run.eval_oracle = _load_policy_eval(d, "evaluate_test_oracle.json", "oracle")

    # Load policies.
    run.candidate_policy = _load_json(d / "candidate_policy.json")
    run.sham_policy = _load_json(d / "sham_policy.json")
    run.candidate_policy_sha256 = _sha256_file(d / "candidate_policy.json")
    run.sham_policy_sha256 = _sha256_file(d / "sham_policy.json")

    # Load training metadata.
    run.candidate_training = _load_json(d / "candidate_training.json")
    run.sham_training = _load_json(d / "sham_training.json")

    # Load Gate A result.
    run.gate_a_result = _load_json(d / "gate_a_result.json")
    run.gate_a_result_sha256 = _sha256_file(d / "gate_a_result.json")

    # Load qualification report.
    run.qualification_report = _load_json(d / "qualification_report.json")
    run.qualification_report_sha256 = _sha256_file(d / "qualification_report.json")

    # Load dataset manifest and examples.
    run.dataset_manifest = _load_json(d / "dataset_manifest.json")
    run.train_examples = _load_list(d / "dataset_train.json")
    run.validation_examples = _load_list(d / "dataset_validation.json")
    run.test_examples = _load_list(d / "dataset_test.json")
    run.ood_examples = _load_list(d / "dataset_ood.json")
    run.safety_examples = _load_list(d / "dataset_safety.json")

    # Derive vocabularies.
    run.all_actions = sorted({o.action_id for o in run.outcomes})
    run.all_families = sorted({t.family for t in run.tasks.values()})
    run.all_archetypes = sorted({t.archetype for t in run.tasks.values()})
    run.all_splits = sorted({t.split for t in run.tasks.values()})

    return run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | list:
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return json.loads(path.read_text())


def _load_list(path: Path) -> list:
    """Load a JSON file that is expected to be a list."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("examples", [])
    return []


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_policy_eval(d: Path, filename: str, policy_id: str) -> PolicyEvaluation | None:
    path = d / filename
    if not path.exists():
        return None
    raw = _load_json(path)
    per_task = {}
    metrics = raw.get("metrics", {})
    # Extract per-task results from metrics.
    task_results = metrics.get("task_results", [])
    for tr in task_results:
        tid = tr.get("task_id", "")
        if tid:
            per_task[tid] = tr
    return PolicyEvaluation(
        policy_id=policy_id,
        split=raw.get("split_name", "test"),
        mean_utility=float(raw.get("mean_utility", 0.0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        n_complete_tasks=int(raw.get("n_complete_tasks", 0)),
        n_manifest_tasks=int(raw.get("n_manifest_tasks", 0)),
        n_missing_tasks=int(raw.get("n_missing_tasks", 0)),
        per_task=per_task,
        metrics=metrics,
        raw_artifact=raw,
    )
