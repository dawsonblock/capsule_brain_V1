"""Split access audit for v0.4.6.

Audits a split access log that records each stage's access to the
immutable data splits.  Every stage must record its split access.

Rules:
    - Training stages → experience split only
    - Calibration stages → validation split only
    - No test access before the final test stage
    - PASS: valid access log proves no forbidden access
    - FAIL: forbidden access occurred
    - BLOCKED: access log absent or incomplete
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus

# Required fields on every access record.
REQUIRED_RECORD_FIELDS = (
    "stage",
    "split",
    "operation",
)

# Stage name markers for classification.
TRAINING_STAGE_MARKERS = (
    "train",
    "fit",
    "learn",
    "candidate",
    "sham",
)

CALIBRATION_STAGE_MARKERS = (
    "calibrate",
    "calibration",
    "threshold",
    "tune",
    "select",
)

TEST_STAGE_MARKERS = (
    "test",
    "gate_a",
    "evaluate",
    "final",
)

# Splits that are forbidden before the final test stage.
TEST_SPLITS = frozenset({"test", "test_outcomes", "test_labels"})


def _stage_lower(stage: str) -> str:
    return (stage or "").lower()


def _is_training_stage(stage: str) -> bool:
    s = _stage_lower(stage)
    return any(m in s for m in TRAINING_STAGE_MARKERS)


def _is_calibration_stage(stage: str) -> bool:
    s = _stage_lower(stage)
    return any(m in s for m in CALIBRATION_STAGE_MARKERS)


def _is_test_stage(stage: str) -> bool:
    s = _stage_lower(stage)
    return any(m in s for m in TEST_STAGE_MARKERS)


def _missing_fields(record: dict) -> list[str]:
    return [
        f for f in REQUIRED_RECORD_FIELDS
        if f not in record or record[f] in (None, "")
    ]


def audit_split_access(
    access_log: list[dict] | None,
    split_manifest: dict,
) -> dict:
    """Audit a split access log for forbidden access patterns.

    Parameters
    ----------
    access_log:
        A list of access records. Each record must include ``stage``,
        ``split``, and ``operation``.  If None or empty, the audit is
        BLOCKED.
    split_manifest:
        The split manifest dict, used to determine valid split names
        and task assignments.

    Returns
    -------
    dict
        status (PASS/FAIL/BLOCKED), forbidden_accesses, n_stages_audited,
        checks, reason
    """
    checks: list[dict[str, Any]] = []
    forbidden_accesses: list[dict[str, Any]] = []

    # --- BLOCKED: access log absent ---
    if access_log is None:
        return {
            "status": AuditStatus.BLOCKED.value,
            "forbidden_accesses": [],
            "n_stages_audited": 0,
            "checks": [],
            "reason": "split access log is absent",
        }

    if not isinstance(access_log, list) or len(access_log) == 0:
        return {
            "status": AuditStatus.BLOCKED.value,
            "forbidden_accesses": [],
            "n_stages_audited": 0,
            "checks": [],
            "reason": "split access log is empty or not a list",
        }

    # --- Determine valid splits from manifest ---
    valid_splits: set[str] = set()
    if isinstance(split_manifest, dict):
        splits = split_manifest.get("splits", {})
        if isinstance(splits, dict):
            valid_splits = set(splits.keys())
    if not valid_splits:
        valid_splits = {"experience", "validation", "test", "ood", "safety"}

    # --- Check structural completeness ---
    malformed: list[dict[str, Any]] = []
    for idx, rec in enumerate(access_log):
        if not isinstance(rec, dict):
            malformed.append({"index": idx, "reason": "record is not a dict"})
            continue
        missing = _missing_fields(rec)
        if missing:
            malformed.append({"index": idx, "missing": missing})

    checks.append({
        "check": "records_structurally_complete",
        "status": "PASS" if not malformed else "FAIL",
        "observed": len(malformed),
        "expected": 0,
        "reason": "every record must carry stage, split, operation",
    })

    if malformed:
        return {
            "status": AuditStatus.BLOCKED.value,
            "forbidden_accesses": [],
            "n_stages_audited": 0,
            "checks": checks,
            "reason": (
                f"access log incomplete: {len(malformed)} record(s) "
                f"missing required fields"
            ),
        }

    # --- Only consider well-formed records ---
    valid_records = [
        rec for rec in access_log
        if isinstance(rec, dict) and not _missing_fields(rec)
    ]

    stages_seen: set[str] = set()
    for rec in valid_records:
        stages_seen.add(rec["stage"])
    n_stages = len(stages_seen)

    # --- Check training stages only access experience ---
    training_violations: list[dict[str, Any]] = []
    for rec in valid_records:
        if not _is_training_stage(rec["stage"]):
            continue
        split = (rec.get("split") or "").lower()
        if split != "experience" and split in valid_splits:
            training_violations.append({
                "stage": rec["stage"],
                "split": rec["split"],
                "operation": rec.get("operation", ""),
                "reason": f"training stage accessed non-experience split '{split}'",
            })
    forbidden_accesses.extend(training_violations)
    checks.append({
        "check": "training_stages_use_experience_only",
        "status": "PASS" if not training_violations else "FAIL",
        "observed": len(training_violations),
        "expected": 0,
        "reason": "training stages may only access the experience split",
    })

    # --- Check calibration stages only access validation ---
    calibration_violations: list[dict[str, Any]] = []
    for rec in valid_records:
        if not _is_calibration_stage(rec["stage"]):
            continue
        split = (rec.get("split") or "").lower()
        if split != "validation" and split in valid_splits:
            calibration_violations.append({
                "stage": rec["stage"],
                "split": rec["split"],
                "operation": rec.get("operation", ""),
                "reason": f"calibration stage accessed non-validation split '{split}'",
            })
    forbidden_accesses.extend(calibration_violations)
    checks.append({
        "check": "calibration_stages_use_validation_only",
        "status": "PASS" if not calibration_violations else "FAIL",
        "observed": len(calibration_violations),
        "expected": 0,
        "reason": "calibration stages may only access the validation split",
    })

    # --- Check no test access before final test stage ---
    # Find the first test-stage record.
    first_test_stage_idx: int | None = None
    for idx, rec in enumerate(valid_records):
        if _is_test_stage(rec["stage"]):
            first_test_stage_idx = idx
            break

    # Any non-test-stage record accessing test splits before the test
    # stage is a violation.
    early_test_violations: list[dict[str, Any]] = []
    for idx, rec in enumerate(valid_records):
        split = (rec.get("split") or "").lower()
        if split not in TEST_SPLITS:
            continue
        if first_test_stage_idx is not None and idx >= first_test_stage_idx:
            continue
        if _is_test_stage(rec["stage"]):
            continue
        early_test_violations.append({
            "stage": rec["stage"],
            "split": rec["split"],
            "operation": rec.get("operation", ""),
            "reason": "test split accessed before the final test stage",
        })
    forbidden_accesses.extend(early_test_violations)
    checks.append({
        "check": "no_test_access_before_final_test_stage",
        "status": "PASS" if not early_test_violations else "FAIL",
        "observed": len(early_test_violations),
        "expected": 0,
        "reason": "no test split access before the final test stage",
    })

    # --- Determine final status ---
    if forbidden_accesses:
        return {
            "status": AuditStatus.FAIL.value,
            "forbidden_accesses": forbidden_accesses,
            "n_stages_audited": n_stages,
            "checks": checks,
            "reason": (
                f"{len(forbidden_accesses)} forbidden access(es) detected"
            ),
        }

    return {
        "status": AuditStatus.PASS.value,
        "forbidden_accesses": [],
        "n_stages_audited": n_stages,
        "checks": checks,
        "reason": (
            f"valid access log proves no forbidden access across "
            f"{n_stages} stage(s)"
        ),
    }
