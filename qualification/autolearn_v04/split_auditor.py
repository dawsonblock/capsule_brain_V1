"""Split access auditor for strict D_test isolation (Section 7).

Every qualification stage must record which task IDs and splits it loaded.
The final report must verify that D_test is first accessed by
evaluate_gate_a_test. Any earlier access invalidates the qualification.

Immutable splits:
    D_experience  — training data
    D_validation  — threshold tuning, trust-region, collapse diagnostics
    D_test        — final Gate A evaluation ONLY
    D_ood         — robustness evaluation
    D_safety      — safety evaluation

The following must NEVER access D_test before final Gate A evaluation:
    - learner fitting
    - feature selection
    - calibration
    - confidence-threshold tuning
    - shift-threshold tuning
    - trust-region diagnostics
    - action-collapse checks
    - model selection
    - checkpoint selection
    - utility tuning
    - prompt tuning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import read_json, write_json


@dataclass(slots=True)
class SplitAccessRecord:
    stage_name: str
    task_ids_loaded: list[str]
    splits_loaded: list[str]
    access_reason: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "task_ids_loaded": list(self.task_ids_loaded),
            "splits_loaded": list(self.splits_loaded),
            "access_reason": self.access_reason,
            "timestamp": self.timestamp,
        }


class SplitAccessLog:
    """Accumulates split access records and validates D_test isolation."""

    # Stages that are allowed to access D_test.
    _TEST_ALLOWED_STAGES = frozenset({
        "evaluate_gate_a_test",
        "evaluate_gate_a_ood",  # OOD is separate but also post-training
        "evaluate_gate_a_safety",
        "run_post_promotion",
        "evaluate_gate_b_test",
        "report",
        "provenance",
        "run_promotion",
    })

    # Stages that must NEVER access D_test.
    _TEST_FORBIDDEN_STAGES = frozenset({
        "train_candidate",
        "train_sham",
        "build_dataset",
        "diagnose_runtime_completion",
        "trust_region_check",
        "action_collapse_check",
        "feature_selection",
        "calibration",
        "threshold_tuning",
        "model_selection",
        "checkpoint_selection",
    })

    def __init__(self) -> None:
        self._records: list[SplitAccessRecord] = []

    def record(
        self,
        stage_name: str,
        task_ids: list[str],
        splits: list[str],
        reason: str = "",
    ) -> None:
        import datetime
        rec = SplitAccessRecord(
            stage_name=stage_name,
            task_ids_loaded=list(task_ids),
            splits_loaded=list(splits),
            access_reason=reason,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self._records.append(rec)

    def validate(self) -> list[str]:
        """Validate that D_test was not accessed before final evaluation.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []
        test_first_accessed: str | None = None
        for rec in self._records:
            if "test" in rec.splits_loaded:
                if test_first_accessed is None:
                    test_first_accessed = rec.stage_name
                # Check if a forbidden stage accessed D_test.
                if rec.stage_name in self._TEST_FORBIDDEN_STAGES:
                    errors.append(
                        f"Stage '{rec.stage_name}' accessed D_test — "
                        f"this stage must never load test data. "
                        f"Reason: {rec.access_reason}"
                    )
        # D_test must first be accessed by an allowed stage.
        if test_first_accessed is not None:
            if test_first_accessed not in self._TEST_ALLOWED_STAGES:
                errors.append(
                    f"D_test was first accessed by '{test_first_accessed}', "
                    f"which is not in the allowed list. "
                    f"First access must be by evaluate_gate_a_test or similar."
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "split-access-log/1",
            "records": [r.to_dict() for r in self._records],
            "validation_errors": self.validate(),
            "valid": len(self.validate()) == 0,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SplitAccessLog":
        data = read_json(path)
        log = cls()
        for rec_data in data.get("records", []):
            log._records.append(SplitAccessRecord(
                stage_name=rec_data["stage_name"],
                task_ids_loaded=rec_data.get("task_ids_loaded", []),
                splits_loaded=rec_data.get("splits_loaded", []),
                access_reason=rec_data.get("access_reason", ""),
                timestamp=rec_data.get("timestamp", ""),
            ))
        return log


# Global singleton for convenience.
_global_log: SplitAccessLog | None = None


def get_global_access_log() -> SplitAccessLog:
    global _global_log
    if _global_log is None:
        _global_log = SplitAccessLog()
    return _global_log


def reset_global_access_log() -> None:
    global _global_log
    _global_log = SplitAccessLog()
