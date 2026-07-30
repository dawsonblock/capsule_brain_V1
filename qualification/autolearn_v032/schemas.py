"""Schemas and JSON I/O helpers for AutoLearn v0.3.2 qualification artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical JSON / hashing helpers
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return sha256_bytes(p.read_bytes())


def write_json(path: str | Path, obj: Any) -> str:
    """Write pretty JSON and return its SHA-256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    p.write_text(text, encoding="utf-8")
    return sha256_text(text)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Benchmark task manifest entry
# ---------------------------------------------------------------------------


# The four families v0.3.2 provisions through real services, plus safety.
FAMILIES = (
    "direct_answer",
    "memory_required",
    "tool_required",
    "workflow_required",
    "safety_adversarial",
)

SPLITS = ("experience", "validation", "test", "ood", "safety")

# Crossover archetypes — cases where the obvious routing proxy is misleading.
CROSSOVER_TYPES = (
    "memory_exists_direct_optimal",
    "tool_exists_direct_optimal",
    "tool_mentioned_no_tool_required",
    "workflow_capability_wasteful",
    "memory_conflict_retrieval_unsafe",
    "tool_failure_direct_better",
    "workflow_failure_tool_better",
    "same_archetype_different_setup",
)


@dataclass(slots=True)
class BenchmarkTask:
    """A single benchmark task definition (manifest entry)."""

    task_id: str
    family: str
    archetype: str
    split: str
    prompt: str
    allowed_actions: tuple[str, ...]
    setup_spec: dict[str, Any]
    verifier_spec: dict[str, Any]
    expected_output_digest: str = ""
    generator_seed: int = 0
    risk_class: str = "standard"
    crossover_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "archetype": self.archetype,
            "split": self.split,
            "prompt": self.prompt,
            "allowed_actions": list(self.allowed_actions),
            "setup_spec": self.setup_spec,
            "verifier_spec": self.verifier_spec,
            "expected_output_digest": self.expected_output_digest,
            "generator_seed": self.generator_seed,
            "risk_class": self.risk_class,
            "crossover_type": self.crossover_type,
        }


def task_to_manifest_entry(task: BenchmarkTask) -> dict[str, Any]:
    """Manifest entry excludes hidden answers from model-visible fields.

    The prompt, allowed_actions, and setup_spec's model-visible subset are
    recorded. The verifier_spec holds the ground truth (expected secret/tool
    output/acceptance contract) and is stored ONLY in the manifest, never
    handed to the model.
    """
    return task.to_dict()


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitEntry:
    task_id: str
    family: str
    archetype: str
    split: str
    allowed_actions: tuple[str, ...]
    verifier_type: str
    setup_digest: str
    prompt_digest: str
    crossover_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "archetype": self.archetype,
            "split": self.split,
            "allowed_actions": list(self.allowed_actions),
            "verifier_type": self.verifier_type,
            "setup_digest": self.setup_digest,
            "prompt_digest": self.prompt_digest,
            "crossover_type": self.crossover_type,
        }


def split_entry_from_task(task: BenchmarkTask) -> SplitEntry:
    return SplitEntry(
        task_id=task.task_id,
        family=task.family,
        archetype=task.archetype,
        split=task.split,
        allowed_actions=task.allowed_actions,
        verifier_type=task.verifier_spec.get("type", "deterministic"),
        setup_digest=sha256_json(task.setup_spec),
        prompt_digest=sha256_text(task.prompt),
        crossover_type=task.crossover_type,
    )
