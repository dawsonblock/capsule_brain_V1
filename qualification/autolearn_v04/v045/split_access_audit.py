"""Split access audit for v0.4.5 final-test leakage control.

Audits a ``split_access_log`` that records each stage's access to the
immutable data splits. The audit enforces strict D_test isolation:

* Pre-Gate-A stages may access ``experience``, ``validation`` and the
  metadata-only test manifest. They may NOT access test outcomes or
  labels.
* The Gate A test stage may access ``test`` exactly once.
* Post-Gate-A analysis must be marked retrospective.

Each access record must include:
    stage, artifact, split, operation, timestamp, run_id
"""
from __future__ import annotations

from typing import Any

# Required fields on every access record.
REQUIRED_RECORD_FIELDS = (
    "stage",
    "artifact",
    "split",
    "operation",
    "timestamp",
    "run_id",
)

# Splits a pre-Gate-A stage is permitted to touch.
PRE_GATE_A_ALLOWED_SPLITS = frozenset({
    "experience",
    "validation",
    "test_manifest_metadata",
})

# Splits a pre-Gate-A stage must never touch (test outcomes / labels).
PRE_GATE_A_FORBIDDEN_SPLITS = frozenset({
    "test",
    "test_outcomes",
    "test_labels",
})

# Stage name markers used to classify access records.
GATE_A_TEST_STAGE_MARKERS = (
    "gate_a_test",
    "evaluate_gate_a_test",
    "gate_a",
)

PRE_GATE_A_STAGE_MARKERS = (
    "train",
    "fit",
    "select",
    "tune",
    "calibrate",
    "threshold",
    "build",
    "feature",
    "sham",
    "candidate",
    "trust_region",
    "collapse",
    "checkpoint",
    "model_selection",
    "prompt",
    "utility",
)

POST_GATE_A_STAGE_MARKERS = (
    "post_gate_a",
    "post_promotion",
    "retrospective",
    "analysis",
    "report",
    "provenance",
    "gate_b",
)


def _is_pre_gate_a(stage: str) -> bool:
    s = (stage or "").lower()
    if any(m in s for m in GATE_A_TEST_STAGE_MARKERS):
        return False
    if any(m in s for m in POST_GATE_A_STAGE_MARKERS):
        return False
    return any(m in s for m in PRE_GATE_A_STAGE_MARKERS)


def _is_gate_a_test_stage(stage: str) -> bool:
    s = (stage or "").lower()
    # Post-Gate-A stages (e.g. "post_gate_a_analysis") contain the
    # "gate_a" substring but are NOT the Gate A test stage itself.
    if _is_post_gate_a(stage):
        return False
    return any(m in s for m in GATE_A_TEST_STAGE_MARKERS)


def _is_post_gate_a(stage: str) -> bool:
    s = (stage or "").lower()
    return any(m in s for m in POST_GATE_A_STAGE_MARKERS)


def _missing_fields(record: dict) -> list[str]:
    return [f for f in REQUIRED_RECORD_FIELDS if f not in record or record[f] in (None, "")]


def audit_split_access(access_log: list[dict] | None) -> dict:
    """Audit a split access log for final-test leakage.

    Parameters
    ----------
    access_log:
        A list of access records. Each record must include the fields
        ``stage``, ``artifact``, ``split``, ``operation``, ``timestamp``
        and ``run_id``.

    Returns
    -------
    dict
        ``status`` (``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"``),
        ``forbidden_accesses``, ``n_stages_audited``, ``checks`` and
        ``errors``.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    forbidden_accesses: list[dict[str, Any]] = []

    # Missing / empty log -> BLOCKED.
    if access_log is None or (isinstance(access_log, list) and len(access_log) == 0):
        return {
            "status": "BLOCKED",
            "forbidden_accesses": [],
            "n_stages_audited": 0,
            "checks": [],
            "errors": ["split access log missing"],
        }

    if not isinstance(access_log, list):
        return {
            "status": "BLOCKED",
            "forbidden_accesses": [],
            "n_stages_audited": 0,
            "checks": [],
            "errors": ["split access log must be a list of records"],
        }

    # Structural completeness: every record must carry the required fields.
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
        "reason": "every record must carry stage, artifact, split, operation, timestamp, run_id",
    })
    if malformed:
        errors.append(f"{len(malformed)} access records missing required fields")

    # Only well-formed records are considered for the leakage checks.
    valid_records = [
        rec for rec in access_log
        if isinstance(rec, dict) and not _missing_fields(rec)
    ]

    stages_seen: set[str] = set()
    for rec in valid_records:
        stages_seen.add(rec["stage"])

    # Pre-Gate-A stages must not touch test outcomes / labels.
    pre_gate_forbidden: list[dict[str, Any]] = []
    for rec in valid_records:
        if not _is_pre_gate_a(rec["stage"]):
            continue
        split = (rec.get("split") or "").lower()
        if split in PRE_GATE_A_FORBIDDEN_SPLITS or split == "test":
            pre_gate_forbidden.append({
                "stage": rec["stage"],
                "artifact": rec["artifact"],
                "split": rec["split"],
                "operation": rec["operation"],
                "run_id": rec["run_id"],
                "reason": "pre-Gate-A stage accessed test outcomes/labels",
            })
    forbidden_accesses.extend(pre_gate_forbidden)
    checks.append({
        "check": "pre_gate_a_no_test_outcomes",
        "status": "PASS" if not pre_gate_forbidden else "FAIL",
        "observed": len(pre_gate_forbidden),
        "expected": 0,
        "reason": "pre-Gate-A stages may not access test outcomes or labels",
    })
    if pre_gate_forbidden:
        errors.append(
            f"{len(pre_gate_forbidden)} pre-Gate-A accesses to test outcomes/labels"
        )

    # Gate A test stage may access test exactly once.
    gate_a_test_accesses = [
        rec for rec in valid_records
        if _is_gate_a_test_stage(rec["stage"])
        and (rec.get("split") or "").lower() in ("test", "test_outcomes", "test_labels")
    ]
    n_gate_a_test = len(gate_a_test_accesses)
    checks.append({
        "check": "gate_a_test_accessed_once",
        "status": "PASS" if n_gate_a_test == 1 else "FAIL",
        "observed": n_gate_a_test,
        "expected": 1,
        "reason": "the Gate A test stage may access test exactly once",
    })
    if n_gate_a_test != 1:
        errors.append(
            f"Gate A test stage accessed test {n_gate_a_test} time(s); expected exactly 1"
        )

    # Test must be first accessed by the Gate A test stage.
    first_test_access: dict[str, Any] | None = None
    for rec in valid_records:
        if (rec.get("split") or "").lower() in ("test", "test_outcomes", "test_labels"):
            first_test_access = rec
            break
    first_test_ok = (
        first_test_access is not None
        and _is_gate_a_test_stage(first_test_access["stage"])
    )
    checks.append({
        "check": "test_first_accessed_by_gate_a",
        "status": "PASS" if first_test_ok else "FAIL",
        "observed": first_test_access["stage"] if first_test_access else "none",
        "expected": "gate_a_test stage",
        "reason": "D_test must first be accessed by the Gate A test stage",
    })
    if not first_test_ok:
        errors.append("D_test was not first accessed by the Gate A test stage")

    # Post-Gate-A analysis must be marked retrospective.
    non_retrospective_post: list[dict[str, Any]] = []
    for rec in valid_records:
        if not _is_post_gate_a(rec["stage"]):
            continue
        op = (rec.get("operation") or "").lower()
        artifact = (rec.get("artifact") or "").lower()
        stage = (rec.get("stage") or "").lower()
        retrospective = (
            "retrospective" in op
            or "retrospective" in artifact
            or "retrospective" in stage
        )
        if not retrospective:
            non_retrospective_post.append({
                "stage": rec["stage"],
                "artifact": rec["artifact"],
                "operation": rec["operation"],
                "reason": "post-Gate-A analysis not marked retrospective",
            })
    checks.append({
        "check": "post_gate_a_marked_retrospective",
        "status": "PASS" if not non_retrospective_post else "FAIL",
        "observed": len(non_retrospective_post),
        "expected": 0,
        "reason": "post-Gate-A analysis must be marked retrospective",
    })
    if non_retrospective_post:
        errors.append(
            f"{len(non_retrospective_post)} post-Gate-A accesses not marked retrospective"
        )

    # No pre-Gate-A stage may access test at all (defensive duplicate of the
    # outcome check, but expressed over the split value directly).
    pre_test_any = [
        rec for rec in valid_records
        if _is_pre_gate_a(rec["stage"])
        and (rec.get("split") or "").lower() == "test"
    ]
    checks.append({
        "check": "pre_gate_a_no_test_split",
        "status": "PASS" if not pre_test_any else "FAIL",
        "observed": len(pre_test_any),
        "expected": 0,
        "reason": "pre-Gate-A stages may only access experience, validation, metadata-only test manifest",
    })
    if pre_test_any:
        for rec in pre_test_any:
            forbidden_accesses.append({
                "stage": rec["stage"],
                "artifact": rec["artifact"],
                "split": rec["split"],
                "operation": rec["operation"],
                "run_id": rec["run_id"],
                "reason": "pre-Gate-A stage accessed test split",
            })
        errors.append(
            f"{len(pre_test_any)} pre-Gate-A accesses to the test split"
        )

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    status = "PASS" if n_fail == 0 else "FAIL"

    return {
        "status": status,
        "forbidden_accesses": forbidden_accesses,
        "n_stages_audited": len(stages_seen),
        "checks": checks,
        "errors": errors,
    }
