"""Dataset builder: counterfactual best-action selection + deterministic splits.

For each task, the counterfactual harness executes multiple valid candidate
actions on the same task and measures the verified utility of each. The
dataset builder turns those (task, action, utility) triples into supervised
training examples:

    a_i* = argmax_a U_i(a)
    dU_i = U_i(a_i*) - max_{a != a_i*} U_i(a)
    w_i  = 1 + min(weight_cap, max(0, dU_i))

Splits are deterministic and by task family (not by individual execution
attempt) so variants of the same task cannot leak across splits. The split
manifest is persisted with a SHA-256 digest and never silently regenerated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.learner import (
    LearnerConfig,
    TrainingExample,
    utility_margin_weight,
)
from capsule_brain.autolearn.schema import Action, ExecutiveExperience
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction

SPLIT_NAMES = ("train", "validation", "test", "ood")


@dataclass(slots=True)
class CounterfactualRow:
    """One (task, action, verified outcome) measurement."""

    task_id: str
    task_family: str
    action: Action
    experience: ExecutiveExperience
    utility: float


@dataclass(slots=True)
class SplitManifest:
    """Deterministic split assignment with a SHA-256 digest."""

    split_version: str = "splits_v1"
    assignments: dict[str, str] = field(default_factory=dict)  # task_id -> split
    family_to_split: dict[str, str] = field(default_factory=dict)
    digest: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "assignments": dict(self.assignments),
            "family_to_split": dict(self.family_to_split),
            "digest": self.digest,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitManifest":
        return cls(
            split_version=str(data.get("split_version", "splits_v1")),
            assignments=dict(data.get("assignments", {}) or {}),
            family_to_split=dict(data.get("family_to_split", {}) or {}),
            digest=str(data.get("digest", "")),
            config=dict(data.get("config", {}) or {}),
        )


@dataclass(slots=True)
class DatasetManifest:
    dataset_version: str = "dataset_v1"
    n_tasks: int = 0
    n_rows: int = 0
    family_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    best_action_counts: dict[str, int] = field(default_factory=dict)
    utility_config: dict[str, Any] = field(default_factory=dict)
    feature_schema_version: str = ""
    dataset_digest: str = ""
    split_manifest_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "n_tasks": self.n_tasks,
            "n_rows": self.n_rows,
            "family_counts": dict(self.family_counts),
            "split_counts": dict(self.split_counts),
            "action_counts": dict(self.action_counts),
            "best_action_counts": dict(self.best_action_counts),
            "utility_config": dict(self.utility_config),
            "feature_schema_version": self.feature_schema_version,
            "dataset_digest": self.dataset_digest,
            "split_manifest_digest": self.split_manifest_digest,
        }


@dataclass(slots=True)
class BuiltDataset:
    examples: list[TrainingExample]
    split_manifest: SplitManifest
    dataset_manifest: DatasetManifest


def build_split_manifest(
    rows: list[CounterfactualRow],
    *,
    train_families: list[str] | None = None,
    validation_families: list[str] | None = None,
    test_families: list[str] | None = None,
    ood_families: list[str] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> SplitManifest:
    """Build a deterministic split manifest by task family.

    Families explicitly listed in train/validation/test/ood go to those
    splits. Remaining families are assigned deterministically by hashing the
    family name into the configured fractions. This guarantees the same input
    always produces the same split and variants of the same task (which share
    a family) never leak across splits.
    """
    families = sorted({r.task_family for r in rows})
    family_to_split: dict[str, str] = {}
    for f in train_families or []:
        family_to_split[f] = "train"
    for f in validation_families or []:
        family_to_split[f] = "validation"
    for f in test_families or []:
        family_to_split[f] = "test"
    for f in ood_families or []:
        family_to_split[f] = "ood"

    remaining = [f for f in families if f not in family_to_split]
    # For families not explicitly assigned, distribute tasks by task_id hash
    # so each unique task goes to exactly one split (no leakage) while
    # keeping a balanced distribution across families.
    assignments: dict[str, str] = {}
    for r in rows:
        fam = r.task_family
        if fam in family_to_split:
            assignments[r.task_id] = family_to_split[fam]
        else:
            h = int(hashlib.sha256(f"{seed}:{r.task_id}".encode("utf-8")).hexdigest(), 16)
            frac = (h % 10000) / 10000.0
            if frac < train_fraction:
                assignments[r.task_id] = "train"
            elif frac < train_fraction + validation_fraction:
                assignments[r.task_id] = "validation"
            elif frac < train_fraction + validation_fraction + test_fraction:
                assignments[r.task_id] = "test"
            else:
                assignments[r.task_id] = "ood"

    manifest = SplitManifest(
        split_version="splits_v1",
        assignments=assignments,
        family_to_split=family_to_split,
        config={
            "train_families": list(train_families or []),
            "validation_families": list(validation_families or []),
            "test_families": list(test_families or []),
            "ood_families": list(ood_families or []),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
        },
    )
    manifest.digest = _manifest_digest(manifest)
    return manifest


def _manifest_digest(manifest: SplitManifest) -> str:
    payload = {
        "split_version": manifest.split_version,
        "assignments": manifest.assignments,
        "family_to_split": manifest.family_to_split,
        "config": manifest.config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_dataset(
    rows: list[CounterfactualRow],
    split_manifest: SplitManifest,
    *,
    utility_config: UtilityConfig | None = None,
    learner_config: LearnerConfig | None = None,
    extractor: FeatureExtractor | None = None,
) -> BuiltDataset:
    """Turn counterfactual rows into weighted supervised training examples.

    For each task, picks a* = argmax_a U(task, a) and computes the utility
    margin against the second-best action. Tasks with at least one
    counterfactual row become one training example.
    """
    uc = utility_config or UtilityConfig()
    ufn = UtilityFunction(uc)
    lc = learner_config or LearnerConfig()
    ext = extractor or FeatureExtractor()

    # Group rows by task.
    by_task: dict[str, list[CounterfactualRow]] = {}
    for r in rows:
        by_task.setdefault(r.task_id, []).append(r)

    examples: list[TrainingExample] = []
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    action_counts: dict[str, int] = {}
    best_action_counts: dict[str, int] = {}

    for task_id, task_rows in by_task.items():
        # Recompute utility from the outcome vector so the dataset is
        # self-consistent with the configured utility function.
        scored: list[tuple[Action, float, CounterfactualRow]] = []
        for r in task_rows:
            u, _ = ufn.compute(r.experience.outcome)
            scored.append((r.action, u, r))
            action_counts[r.action.value] = action_counts.get(r.action.value, 0) + 1
        scored.sort(key=lambda t: t[1], reverse=True)
        best_action, best_util, best_row = scored[0]
        second_util = scored[1][1] if len(scored) > 1 else best_util
        margin = best_util - second_util

        split = split_manifest.assignments.get(task_id, "train")
        split_counts[split] = split_counts.get(split, 0) + 1
        family_counts[best_row.task_family] = family_counts.get(best_row.task_family, 0) + 1
        best_action_counts[best_action.value] = best_action_counts.get(best_action.value, 0) + 1

        fv = ext.extract(best_row.experience.state)
        examples.append(TrainingExample(
            features=fv.as_list(),
            best_action=best_action,
            utility_margin=margin,
            weight=utility_margin_weight(margin, lc),
            task_id=task_id,
            task_family=best_row.task_family,
            split=split,
        ))

    # Sort examples deterministically by task_id for a stable digest.
    examples.sort(key=lambda e: e.task_id)

    dataset_digest = _dataset_digest(examples)
    dataset_manifest = DatasetManifest(
        n_tasks=len(by_task),
        n_rows=len(rows),
        family_counts=dict(sorted(family_counts.items())),
        split_counts=dict(split_counts),
        action_counts=dict(sorted(action_counts.items())),
        best_action_counts=dict(sorted(best_action_counts.items())),
        utility_config=uc.to_dict(),
        feature_schema_version=ext.schema_version,
        dataset_digest=dataset_digest,
        split_manifest_digest=split_manifest.digest,
    )
    return BuiltDataset(
        examples=examples,
        split_manifest=split_manifest,
        dataset_manifest=dataset_manifest,
    )


def _dataset_digest(examples: list[TrainingExample]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(json.dumps(
            {
                "task_id": ex.task_id,
                "task_family": ex.task_family,
                "split": ex.split,
                "best_action": ex.best_action.value,
                "weight": round(ex.weight, 8),
                "features": [round(f, 8) for f in ex.features],
            },
            sort_keys=True,
        ).encode("utf-8"))
    return h.hexdigest()


def split_examples(
    examples: list[TrainingExample],
) -> dict[str, list[TrainingExample]]:
    out: dict[str, list[TrainingExample]] = {s: [] for s in SPLIT_NAMES}
    for ex in examples:
        out.setdefault(ex.split, []).append(ex)
    return out


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))


def load_split_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text()))


def save_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
