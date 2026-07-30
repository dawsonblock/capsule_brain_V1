"""Strict dataset split design for AutoLearn v0.3.1.

v0.3.1 Section 10: Four immutable splits with archetype disjointness.

Splits:
- D_experience: training data
- D_validation: threshold tuning only
- D_test: final evaluation only
- D_ood: out-of-distribution evaluation
- D_safety: adversarial safety tasks

Required disjointness: archetypes must not overlap between any two splits.
No final-test result may be available to training, threshold selection,
calibration, shift tuning, model selection, feature selection, utility
tuning, or promotion-threshold tuning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qualification.autolearn_v031.schemas import SplitEntry


def validate_split_disjointness(entries: list[SplitEntry]) -> list[str]:
    """Validate that all splits are disjoint by archetype and task_id.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check: no task_id appears in multiple splits.
    task_splits: dict[str, str] = {}
    for e in entries:
        if e.task_id in task_splits:
            errors.append(
                f"task_id {e.task_id} appears in both "
                f"{task_splits[e.task_id]} and {e.split}"
            )
        else:
            task_splits[e.task_id] = e.split

    # Check: archetypes are disjoint between splits.
    split_archetypes: dict[str, set[str]] = {}
    for e in entries:
        split_archetypes.setdefault(e.split, set()).add(e.archetype)

    splits = list(split_archetypes.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = split_archetypes[s1] & split_archetypes[s2]
            if overlap:
                errors.append(
                    f"archetype overlap between {s1} and {s2}: {overlap}"
                )

    return errors


def validate_split_sizes(
    entries: list[SplitEntry],
    *,
    min_experience: int = 160,
    min_validation: int = 60,
    min_test: int = 100,
    min_ood: int = 60,
    min_safety: int = 40,
) -> list[str]:
    """Validate that each split has the minimum number of tasks."""
    errors: list[str] = []
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.split] = counts.get(e.split, 0) + 1

    minima = {
        "experience": min_experience,
        "validation": min_validation,
        "test": min_test,
        "ood": min_ood,
        "safety": min_safety,
    }
    for split, minimum in minima.items():
        actual = counts.get(split, 0)
        if actual < minimum:
            errors.append(
                f"split '{split}' has {actual} tasks, minimum is {minimum}"
            )

    return errors


def write_split_manifest(
    entries: list[SplitEntry],
    output_path: str | Path,
) -> str:
    """Write the split manifest and return its SHA-256."""
    import hashlib
    data = {
        "splits": [e.to_dict() for e in entries],
        "n_tasks": len(entries),
    }
    text = json.dumps(data, indent=2, sort_keys=True)
    Path(output_path).write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split_manifest(path: str | Path) -> list[SplitEntry]:
    """Load a split manifest from JSON."""
    data = json.loads(Path(path).read_text())
    return [
        SplitEntry(
            task_id=e["task_id"],
            family=e["family"],
            archetype=e["archetype"],
            generator=e["generator"],
            seed=e["seed"],
            split=e["split"],
            allowed_actions=tuple(e["allowed_actions"]),
            verifier_type=e["verifier_type"],
            setup_digest=e["setup_digest"],
            prompt_digest=e["prompt_digest"],
        )
        for e in data.get("splits", [])
    ]


def validate_no_test_leakage(
    training_task_ids: set[str],
    test_task_ids: set[str],
) -> list[str]:
    """Validate that no test task IDs appear in training data."""
    errors: list[str] = []
    overlap = training_task_ids & test_task_ids
    if overlap:
        errors.append(
            f"test data leakage: {len(overlap)} task IDs appear in both "
            f"training and test: {list(overlap)[:5]}..."
        )
    return errors
