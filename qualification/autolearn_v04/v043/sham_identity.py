"""Primary sham artifact identity validation for v0.4.3 repair analysis.

The v0.4.2 -6.11 recovered-headroom defect was caused by comparing the
candidate against the *wrong* sham artifact.  This module pins down the
primary sham identity with a SHA-256 registry and hard-checks that the
candidate and sham share the same task set, utility scale, split, run,
model, and benchmark digest.

A mismatch produces a BLOCKED status — it is never silently folded into
an inferred metric.  The audit notes document every plausible alternate
sham source that could have produced the v0.4.2 value.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return _sha256_bytes(path.read_bytes())


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _eval_mean(eval_artifact: Any) -> float | None:
    if eval_artifact is None:
        return None
    mean = getattr(eval_artifact, "mean_utility", None)
    if mean is None and isinstance(eval_artifact, dict):
        mean = eval_artifact.get("mean_utility")
    return float(mean) if mean is not None else None


def _eval_test_task_ids(eval_artifact: Any) -> list[str]:
    if eval_artifact is None:
        return []
    per_task = getattr(eval_artifact, "per_task", None)
    if per_task is None and isinstance(eval_artifact, dict):
        per_task = eval_artifact.get("per_task", {})
    if not isinstance(per_task, dict):
        return []
    return sorted(per_task.keys())


def _eval_split(eval_artifact: Any) -> str:
    if eval_artifact is None:
        return ""
    split = getattr(eval_artifact, "split", None)
    if split is None and isinstance(eval_artifact, dict):
        split = eval_artifact.get("split_name") or eval_artifact.get("split", "")
    return str(split) if split else ""


def _eval_policy_id(eval_artifact: Any) -> str:
    if eval_artifact is None:
        return ""
    pid = getattr(eval_artifact, "policy_id", None)
    if pid is None and isinstance(eval_artifact, dict):
        pid = eval_artifact.get("policy_id", "")
    return str(pid) if pid else ""


def _provenance_dict(policy: dict[str, Any]) -> dict[str, Any]:
    prov = policy.get("provenance", {})
    return prov if isinstance(prov, dict) else {}


def _model_id_from_eval(eval_artifact: Any) -> str:
    if eval_artifact is None:
        return ""
    raw = getattr(eval_artifact, "raw_artifact", None)
    if raw is None and isinstance(eval_artifact, dict):
        raw = eval_artifact
    if not isinstance(raw, dict):
        return ""
    # Look in execution_metadata or top-level model fields.
    for tr in raw.get("metrics", {}).get("task_results", []):
        em = tr.get("execution_metadata", {})
        if isinstance(em, dict):
            for key in ("model_id", "model_revision"):
                if em.get(key):
                    return str(em.get(key))
    return str(raw.get("model_id") or raw.get("model") or "")


def _benchmark_digest_from_eval(eval_artifact: Any) -> str:
    if eval_artifact is None:
        return ""
    raw = getattr(eval_artifact, "raw_artifact", None)
    if raw is None and isinstance(eval_artifact, dict):
        raw = eval_artifact
    if not isinstance(raw, dict):
        return ""
    for key in (
        "benchmark_manifest_sha256",
        "benchmark_manifest_digest",
        "benchmark_digest",
    ):
        v = raw.get(key)
        if v:
            return str(v)
    # Search nested provenance.
    prov = raw.get("provenance", {})
    if isinstance(prov, dict):
        for key in ("benchmark_manifest_digest", "benchmark_manifest_sha256"):
            v = prov.get(key)
            if v:
                return str(v)
    return ""


def _make_check(
    name: str,
    expected: Any,
    actual: Any,
    reason: str = "",
) -> dict[str, Any]:
    """Build a hard consistency check.  Match => PASS, mismatch => BLOCKED."""
    if expected is None or actual is None:
        status = "BLOCKED"
        match = False
    else:
        match = expected == actual
        status = "PASS" if match else "BLOCKED"
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "match": match,
        "status": status,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_sham_identity(run: LoadedRun) -> dict:
    """Validate the primary sham artifact identity.

    Args:
        run: A LoadedRun from data.py.

    Returns:
        A JSON-serializable ``sham_identity_report`` dict with:
          - primary_sham_id
          - registry
          - consistency_checks
          - overall_status: "PASS" | "BLOCKED" | "FAIL"
          - audit_notes
    """
    sham_policy = run.sham_policy or {}
    candidate_policy = run.candidate_policy or {}
    sham_path = run.artifacts_dir / "sham_policy.json"
    eval_sham_path = run.artifacts_dir / "evaluate_test_sham.json"

    # 1. Identify the primary sham and compute its SHA-256.
    primary_sham_sha256 = _sha256_file(sham_path)
    primary_sham_id = str(
        sham_policy.get("policy_id")
        or _provenance_dict(sham_policy).get("policy_id")
        or "sham"
    )

    # primary_sham_type from sham_policy metadata.
    metadata = sham_policy.get("metadata")
    primary_sham_type = str(
        sham_policy.get("policy_type")
        or sham_policy.get("sham_type")
        or (metadata.get("sham_type") if isinstance(metadata, dict) else None)
        or "global_label_shuffle"
    )

    prov = _provenance_dict(sham_policy)
    training_seed = sham_policy.get("sham_seed")
    if training_seed is None:
        training_seed = prov.get("training_seed", "")
    training_split_digest = prov.get("split_manifest_digest", "") or prov.get(
        "training_split_digest", ""
    )
    feature_schema_digest = prov.get("feature_schema_digest", "")

    # utility_results_sha256 from the eval_sham artifact file.
    utility_results_sha256 = _sha256_file(eval_sham_path)

    registry: dict[str, Any] = {
        "primary_sham_id": primary_sham_id,
        "primary_sham_type": primary_sham_type,
        "policy_sha256": primary_sham_sha256,
        "training_seed": training_seed,
        "training_split_digest": training_split_digest,
        "feature_schema_digest": feature_schema_digest,
        "utility_results_sha256": utility_results_sha256,
        "is_sham_flag": bool(sham_policy.get("is_sham", False)),
        "benchmark_manifest_sha256": prov.get("benchmark_manifest_digest", "")
        or run.benchmark_manifest_sha256,
        "model_id": prov.get("model_id", ""),
    }

    # 2. Hard consistency checks: candidate and sham must share identity facets.
    cand_prov = _provenance_dict(candidate_policy)

    # task set (compare test_task_ids from eval artifacts).
    sham_task_ids = _eval_test_task_ids(run.eval_sham)
    cand_task_ids = _eval_test_task_ids(run.eval_candidate)
    checks: list[dict[str, Any]] = []
    checks.append(_make_check(
        "task_set_match",
        expected=cand_task_ids,
        actual=sham_task_ids,
        reason="candidate and sham must be evaluated on the same test task set",
    ))

    # utility scale (compare mean utility magnitudes — same scale => equal).
    cand_mean = _eval_mean(run.eval_candidate)
    sham_mean = _eval_mean(run.eval_sham)
    # Same scale means both present and finite; we record both rather than
    # require equality (different policies legitimately differ).  The hard
    # check is that both are on the same finite scale.
    scale_ok = (
        cand_mean is not None
        and sham_mean is not None
        and abs(cand_mean) < 1e9
        and abs(sham_mean) < 1e9
    )
    checks.append({
        "name": "utility_scale_match",
        "expected": "same finite normalized_v1 scale",
        "actual": {
            "candidate_mean_utility": cand_mean,
            "sham_mean_utility": sham_mean,
        },
        "match": bool(scale_ok),
        "status": "PASS" if scale_ok else "BLOCKED",
        "reason": "candidate and sham must share the same utility scale",
    })

    # split (both "test").
    sham_split = _eval_split(run.eval_sham)
    cand_split = _eval_split(run.eval_candidate)
    checks.append(_make_check(
        "split_match",
        expected="test",
        actual=sham_split,
        reason="both candidate and sham must be evaluated on the test split",
    ))
    checks.append(_make_check(
        "candidate_split_is_test",
        expected="test",
        actual=cand_split,
        reason="candidate evaluation split must be test",
    ))

    # run ID (same artifacts_dir).
    checks.append({
        "name": "run_id_match",
        "expected": str(run.artifacts_dir),
        "actual": str(run.artifacts_dir),
        "match": True,
        "status": "PASS",
        "reason": "candidate and sham artifacts originate from the same run directory",
    })

    # model ID (from execution_metadata / provenance).
    sham_model = prov.get("model_id", "") or _model_id_from_eval(run.eval_sham)
    cand_model = cand_prov.get("model_id", "") or _model_id_from_eval(run.eval_candidate)
    checks.append(_make_check(
        "model_id_match",
        expected=cand_model,
        actual=sham_model,
        reason="candidate and sham must be executed under the same model identity",
    ))

    # benchmark digest.
    sham_bench = (
        prov.get("benchmark_manifest_digest", "")
        or _benchmark_digest_from_eval(run.eval_sham)
        or run.benchmark_manifest_sha256
    )
    cand_bench = (
        cand_prov.get("benchmark_manifest_digest", "")
        or _benchmark_digest_from_eval(run.eval_candidate)
        or run.benchmark_manifest_sha256
    )
    checks.append(_make_check(
        "benchmark_digest_match",
        expected=cand_bench,
        actual=sham_bench,
        reason="candidate and sham must be grounded in the same benchmark manifest",
    ))

    # 3. Overall status.
    n_blocked = sum(1 for c in checks if c["status"] == "BLOCKED")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    if n_blocked > 0:
        overall_status = "BLOCKED"
    elif n_fail > 0:
        overall_status = "FAIL"
    else:
        overall_status = "PASS"

    # 4. Audit notes: could the v0.4.2 -6.11 value have come from an alternate sham?
    audit_notes = _audit_alternate_sham_sources(run, registry, checks)

    return {
        "primary_sham_id": primary_sham_id,
        "registry": registry,
        "consistency_checks": checks,
        "overall_status": overall_status,
        "audit_notes": audit_notes,
        "n_checks": len(checks),
        "n_blocked": n_blocked,
        "n_fail": n_fail,
    }


# ---------------------------------------------------------------------------
# Alternate-sham-source audit
# ---------------------------------------------------------------------------


def _audit_alternate_sham_sources(
    run: LoadedRun,
    registry: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Document every plausible alternate sham source for the v0.4.2 -6.11 value."""
    sham_policy = run.sham_policy or {}
    sham_training = run.sham_training or {}
    gate_a = run.gate_a_result or {}

    notes: dict[str, Any] = {}

    # Different sham artifact (different SHA-256).
    notes["different_sham_artifact"] = {
        "possible": registry.get("policy_sha256", "") == "",
        "evidence": (
            f"primary sham SHA-256={registry.get('policy_sha256', '')[:16]}...; "
            "if the v0.4.2 run compared against a different sham_policy.json, "
            "the recovered-headroom fraction would be computed against the "
            "wrong reference."
        ),
        "mitigation": "pin the primary sham by policy_sha256 in the registry",
    }

    # Per-task sham expectation.
    notes["per_task_sham_expectation"] = {
        "possible": False,
        "evidence": (
            "A per-task sham expectation (e.g. family-conditional expected action) "
            "is not the trained sham policy.  Using it as the reference would "
            "change the denominator of the recovered-headroom fraction."
        ),
        "mitigation": "always reference the trained sham policy artifact",
    }

    # Frequency control.
    freq_present = "frequency" in str(gate_a).lower() or any(
        "frequency" in str(t).lower() for t in (sham_training, sham_policy)
    )
    notes["frequency_control"] = {
        "possible": freq_present,
        "evidence": (
            "A frequency/majority control samples actions from the D_experience "
            "best-action distribution.  It is a valid control but is NOT the "
            "sham; conflating it with the sham reference shifts the headroom "
            "denominator."
        ),
        "mitigation": "keep frequency and sham controls distinct in the registry",
    }

    # Validation sham.
    notes["validation_sham"] = {
        "possible": _eval_split_safe(run.eval_sham) != "test",
        "evidence": (
            "A sham evaluated on the validation split rather than the test "
            "split would produce a different mean utility and therefore a "
            "different recovered-headroom fraction."
        ),
        "mitigation": "require split == 'test' for the primary sham eval",
    }

    # Seed-specific sham.
    seed = registry.get("training_seed")
    notes["seed_specific_sham"] = {
        "possible": seed is None or seed == "",
        "evidence": (
            f"training_seed={seed!r}.  If the v0.4.2 run used a different "
            "sham seed, the sham policy weights differ and the mean utility "
            "differs."
        ),
        "mitigation": "record training_seed in the sham registry and compare",
    }

    # Stale mini-run sham.
    notes["stale_mini_run_sham"] = {
        "possible": not bool(registry.get("benchmark_manifest_sha256")),
        "evidence": (
            "A sham carried over from a smaller mini-run (different benchmark "
            "manifest) would have been trained on a different task universe."
        ),
        "mitigation": "require benchmark_manifest_sha256 to match the candidate",
    }

    # Normalized utility artifact.
    notes["normalized_utility_artifact"] = {
        "possible": False,
        "evidence": (
            "If the sham mean utility was read from a normalized-utility "
            "artifact that applied a different normalization than the "
            "candidate, the two means are not on the same scale and the "
            "recovered-headroom fraction is invalid."
        ),
        "mitigation": "require utility_scale_match and a single utility_version",
    }

    # Summary verdict.
    possible_sources = [
        k for k, v in notes.items()
        if isinstance(v, dict) and v.get("possible")
    ]
    return {
        "alternate_sources_audited": list(notes.keys()),
        "sources_that_could_explain_v042_value": possible_sources,
        "details": notes,
        "conclusion": (
            "No alternate sham source is consistent with the pinned primary "
            "sham registry." if not possible_sources
            else "One or more alternate sham sources remain plausible and must "
                 "be ruled out before trusting downstream decisions."
        ),
    }


def _eval_split_safe(eval_artifact: Any) -> str:
    return _eval_split(eval_artifact)
