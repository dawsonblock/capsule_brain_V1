"""Canonical evidence-level registry for v0.4.5.

Every artifact loader reports its achieved evidence level.
Every analysis stage declares its required evidence level.
If achieved < required, the stage is BLOCKED.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EvidenceLevel(IntEnum):
    """Evidence level hierarchy. Higher = more granular data."""
    NONE = 0
    AGGREGATE_ONLY = 1
    TASK_LEVEL = 2
    ACTION_LEVEL_COMPLETE = 3


# Map analysis stages to their minimum required evidence levels.
REQUIRED_EVIDENCE_LEVELS: dict[str, EvidenceLevel] = {
    "summary_display": EvidenceLevel.AGGREGATE_ONLY,
    "candidate_vs_baseline_bootstrap": EvidenceLevel.TASK_LEVEL,
    "candidate_vs_sham_bootstrap": EvidenceLevel.TASK_LEVEL,
    "oracle_routing_headroom": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "headroom_concentration": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "high_margin_recovery": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "family_analysis": EvidenceLevel.TASK_LEVEL,
    "archetype_analysis": EvidenceLevel.TASK_LEVEL,
    "confusion_matrix": EvidenceLevel.TASK_LEVEL,
    "raw_vs_executed": EvidenceLevel.TASK_LEVEL,
    "feature_signal": EvidenceLevel.TASK_LEVEL,
    "learner_validation": EvidenceLevel.TASK_LEVEL,
    "candidate_stability": EvidenceLevel.TASK_LEVEL,
    "sham_ensemble": EvidenceLevel.TASK_LEVEL,
    "metric_consistency": EvidenceLevel.TASK_LEVEL,
    "counterfactual_equivalence": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "action_matrix_completeness": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "safety_evidence": EvidenceLevel.TASK_LEVEL,
    "gate_a0": EvidenceLevel.AGGREGATE_ONLY,
    "gate_a1": EvidenceLevel.ACTION_LEVEL_COMPLETE,
    "gate_a2": EvidenceLevel.TASK_LEVEL,
    "gate_a3": EvidenceLevel.TASK_LEVEL,
    "gate_a4": EvidenceLevel.TASK_LEVEL,
    "gate_a5": EvidenceLevel.ACTION_LEVEL_COMPLETE,
}


@dataclass(frozen=True)
class EvidenceLevelAssessment:
    """Result of evidence-level assessment for a stage."""
    stage: str
    required_level: EvidenceLevel
    achieved_level: EvidenceLevel
    status: str  # "PASS" | "BLOCKED"
    missing_evidence: str
    blocking_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "required_level": self.required_level.name,
            "achieved_level": self.achieved_level.name,
            "status": self.status,
            "missing_evidence": self.missing_evidence,
            "blocking_artifact": self.blocking_artifact,
        }


def assess_evidence_level(
    stage: str,
    achieved_level: EvidenceLevel,
    blocking_artifact: str = "",
    missing_evidence: str = "",
) -> EvidenceLevelAssessment:
    """Assess whether achieved evidence level meets the stage's requirement."""
    required = REQUIRED_EVIDENCE_LEVELS.get(stage, EvidenceLevel.AGGREGATE_ONLY)
    if achieved_level < required:
        if not missing_evidence:
            missing_evidence = f"Requires {required.name}, achieved {achieved_level.name}"
        return EvidenceLevelAssessment(
            stage=stage,
            required_level=required,
            achieved_level=achieved_level,
            status="BLOCKED",
            missing_evidence=missing_evidence,
            blocking_artifact=blocking_artifact,
        )
    return EvidenceLevelAssessment(
        stage=stage,
        required_level=required,
        achieved_level=achieved_level,
        status="PASS",
        missing_evidence="",
        blocking_artifact="",
    )


def detect_evidence_level(
    counterfactual_outcomes: list[dict],
    policy_results: dict | None = None,
    split_manifest: dict | None = None,
) -> EvidenceLevel:
    """Detect the achieved evidence level from available artifacts.

    Returns the highest evidence level that the data supports.
    """
    if not counterfactual_outcomes:
        # Check if we have aggregate-only results.
        if policy_results and policy_results.get("mean_utility") is not None:
            return EvidenceLevel.AGGREGATE_ONLY
        return EvidenceLevel.NONE

    # Check if outcomes have task_id and action fields (task-level).
    has_task_ids = all("task_id" in o for o in counterfactual_outcomes if o.get("task_id"))
    if not has_task_ids:
        return EvidenceLevel.AGGREGATE_ONLY

    # Check if we have complete action matrix (action-level).
    if split_manifest:
        task_actions: dict[str, set[str]] = {}
        for o in counterfactual_outcomes:
            tid = o.get("task_id", "")
            action = o.get("eligible_action", o.get("action", ""))
            if tid and action:
                task_actions.setdefault(tid, set()).add(action)

        eligible_actions = set(split_manifest.get("eligible_actions", []))
        if eligible_actions:
            complete = all(
                eligible_actions.issubset(actions)
                for actions in task_actions.values()
            )
            if complete:
                return EvidenceLevel.ACTION_LEVEL_COMPLETE

    # Has task IDs but not complete action matrix.
    return EvidenceLevel.TASK_LEVEL
