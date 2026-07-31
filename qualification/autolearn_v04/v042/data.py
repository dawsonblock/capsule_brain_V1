"""Core data loading for v0.4.2 diagnostics.

Loads the immutable 7B full-run artifacts and provides a unified
data structure for all analysis modules.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """Metadata for a single benchmark task."""
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


@dataclass(frozen=True, slots=True)
class Outcome:
    """A single counterfactual outcome (task, action) pair."""
    task_id: str
    action_id: str
    availability: str
    verification: str
    utility: float
    execution_metadata: dict[str, Any]
    reward_components: dict[str, Any]


@dataclass
class LoadedRun:
    """All loaded artifacts from a scientific run."""
    artifacts_dir: Path
    benchmark_manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    outcomes: list[Outcome] = field(default_factory=list)
    outcomes_by_task: dict[str, list[Outcome]] = field(default_factory=dict)
    outcomes_by_task_action: dict[tuple[str, str], Outcome] = field(default_factory=dict)
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    train_examples: list[dict] = field(default_factory=list)
    validation_examples: list[dict] = field(default_factory=list)
    test_examples: list[dict] = field(default_factory=list)
    ood_examples: list[dict] = field(default_factory=list)
    safety_examples: list[dict] = field(default_factory=list)
    candidate_policy: dict[str, Any] = field(default_factory=dict)
    sham_policy: dict[str, Any] = field(default_factory=dict)
    candidate_training: dict[str, Any] = field(default_factory=dict)
    sham_training: dict[str, Any] = field(default_factory=dict)
    gate_a_result: dict[str, Any] = field(default_factory=dict)
    promotion_result: dict[str, Any] = field(default_factory=dict)
    qualification_report: dict[str, Any] = field(default_factory=dict)
    runtime_diagnostics: dict[str, Any] = field(default_factory=dict)

    # Derived.
    all_actions: list[str] = field(default_factory=list)
    all_families: list[str] = field(default_factory=list)
    all_archetypes: list[str] = field(default_factory=list)
    all_splits: list[str] = field(default_factory=list)

    # Hashes for provenance.
    benchmark_manifest_sha256: str = ""
    split_manifest_sha256: str = ""
    counterfactual_results_sha256: str = ""
    dataset_sha256: str = ""

    @property
    def test_task_ids(self) -> list[str]:
        """Task IDs for the test split."""
        return [t.task_id for t in self.tasks.values() if t.split == "test"]

    @property
    def ood_task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks.values() if t.split == "ood"]

    @property
    def experience_task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks.values() if t.split == "experience"]

    def get_utility_matrix(self, task_ids: list[str] | None = None) -> tuple[list[str], list[str], np.ndarray]:
        """Build a utility matrix [n_tasks, n_actions] for given tasks.

        Returns (task_ids, action_names, utility_matrix).
        Missing (task, action) pairs get utility 0.0.
        """
        if task_ids is None:
            task_ids = list(self.tasks.keys())
        actions = self.all_actions
        action_idx = {a: i for i, a in enumerate(actions)}
        U = np.zeros((len(task_ids), len(actions)))
        for i, tid in enumerate(task_ids):
            for outcome in self.outcomes_by_task.get(tid, []):
                if outcome.action_id in action_idx and outcome.availability == "executed":
                    U[i, action_idx[outcome.action_id]] = outcome.utility
        return task_ids, actions, U

    def get_best_action(self, task_id: str) -> str | None:
        """Get the measured best (highest utility) action for a task."""
        outcomes = self.outcomes_by_task.get(task_id, [])
        executed = [o for o in outcomes if o.availability == "executed" and o.utility is not None]
        if not executed:
            return None
        return max(executed, key=lambda o: o.utility).action_id

    def get_oracle_utility(self, task_id: str) -> float:
        """Get the max measured utility for a task."""
        outcomes = self.outcomes_by_task.get(task_id, [])
        executed = [o for o in outcomes if o.availability == "executed" and o.utility is not None]
        if not executed:
            return 0.0
        return max(o.utility for o in executed)

    def get_utility_for_action(self, task_id: str, action_id: str) -> float | None:
        """Get measured utility for a specific (task, action) pair."""
        o = self.outcomes_by_task_action.get((task_id, action_id))
        if o is None or o.availability != "executed":
            return None
        return o.utility


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()


def load_run(artifacts_dir: str | Path) -> LoadedRun:
    """Load all artifacts from a scientific run directory.

    Raises FileNotFoundError if critical artifacts are missing.
    """
    artifacts = Path(artifacts_dir)
    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts}")

    def _read(name: str) -> dict:
        p = artifacts / name
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    bm = _read("benchmark_manifest.json")
    split = _read("split_manifest.json")
    ds_manifest = _read("dataset_manifest.json")

    run = LoadedRun(
        artifacts_dir=artifacts,
        benchmark_manifest=bm,
        split_manifest=split,
        dataset_manifest=ds_manifest,
    )

    # Parse tasks.
    for t_dict in bm.get("tasks", []):
        ti = TaskInfo(
            task_id=t_dict["task_id"],
            split=t_dict.get("split", "unknown"),
            family=t_dict.get("family", "unknown"),
            archetype=t_dict.get("archetype", "unknown"),
            allowed_actions=t_dict.get("allowed_actions", []),
            verifier_spec=t_dict.get("verifier_spec", {}),
            prompt=t_dict.get("prompt", ""),
            generator_family=t_dict.get("generator_family", ""),
            template_family=t_dict.get("template_family", ""),
            capability_family=t_dict.get("capability_family", ""),
            entity_family=t_dict.get("entity_family", ""),
            risk_class=t_dict.get("risk_class", ""),
            crossover_type=t_dict.get("crossover_type", ""),
            group_id=t_dict.get("group_id", ""),
            generator_seed=t_dict.get("generator_seed", 0),
            setup_spec=t_dict.get("setup_spec", {}),
            expected_output_digest=t_dict.get("expected_output_digest", ""),
        )
        run.tasks[ti.task_id] = ti

    # Parse outcomes.
    outcomes_raw = _read("counterfactual_outcomes.json")
    if not outcomes_raw:
        # Try legacy format.
        outcomes_raw = _read("real_counterfactual_results.json")
    for o_dict in outcomes_raw:
        o = Outcome(
            task_id=o_dict["task_id"],
            action_id=o_dict["action_id"],
            availability=o_dict.get("availability", "executed"),
            verification=o_dict.get("verification", "unknown"),
            utility=o_dict.get("utility", 0.0),
            execution_metadata=o_dict.get("execution_metadata", {}),
            reward_components=o_dict.get("reward_components", {}),
        )
        run.outcomes.append(o)
        run.outcomes_by_task.setdefault(o.task_id, []).append(o)
        run.outcomes_by_task_action[(o.task_id, o.action_id)] = o

    # Parse dataset splits.
    run.train_examples = _read("dataset_train.json")
    run.validation_examples = _read("dataset_validation.json")
    run.test_examples = _read("dataset_test.json")
    run.ood_examples = _read("dataset_ood.json")
    run.safety_examples = _read("dataset_safety.json")

    # Parse policies and results.
    run.candidate_policy = _read("candidate_policy.json")
    run.sham_policy = _read("sham_policy.json")
    run.candidate_training = _read("candidate_training.json")
    run.sham_training = _read("sham_training.json")
    run.gate_a_result = _read("gate_a_result.json")
    run.promotion_result = _read("promotion_result.json")
    run.qualification_report = _read("qualification_report.json")
    run.runtime_diagnostics = _read("runtime_completion_diagnostics.json")

    # Derive action/family/archetype/split sets.
    action_set: set[str] = set()
    for t in run.tasks.values():
        action_set.update(t.allowed_actions)
    run.all_actions = sorted(action_set)

    run.all_families = sorted(set(t.family for t in run.tasks.values()))
    run.all_archetypes = sorted(set(t.archetype for t in run.tasks.values()))
    run.all_splits = sorted(set(t.split for t in run.tasks.values()))

    # Compute hashes.
    run.benchmark_manifest_sha256 = _sha256_json(bm)
    run.split_manifest_sha256 = _sha256_json(split)
    cf_path = artifacts / "counterfactual_outcomes.json"
    if cf_path.exists():
        run.counterfactual_results_sha256 = _sha256_file(cf_path)
    ds_path = artifacts / "dataset_manifest.json"
    if ds_path.exists():
        run.dataset_sha256 = _sha256_file(ds_path)

    return run


def load_two_runs(mini_dir: str | Path, full_dir: str | Path) -> tuple[LoadedRun, LoadedRun]:
    """Load both the mini and full 7B runs for cross-run comparison."""
    return load_run(mini_dir), load_run(full_dir)
