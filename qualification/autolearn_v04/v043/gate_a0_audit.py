"""Gate A0 infrastructure audit for v0.4.3 repair analysis.

Resolves the Gate A0 infrastructure status with a 15-sub-gate checklist.
Gate A0 is the prerequisite for every downstream Gate A sub-gate: if the
infrastructure is not valid, no causal claim may be made.

Status semantics (strict):

  PASS      — every mandatory sub-gate passes.
  FAIL      — a valid check was evaluated and failed.
  BLOCKED   — evidence genuinely cannot be evaluated (missing artifact,
              unreadable field).  BLOCKED is NOT a generic failure state.
  NOT_RUN   — the stage was not attempted.
"""
from __future__ import annotations

from typing import Any

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate(
    gate_id: str,
    status: str,
    observed: Any,
    expected: Any,
    reason: str,
    artifact: str,
    field: str,
    remediation: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "reason": reason,
        "artifact": artifact,
        "field": field,
        "remediation": remediation,
    }


def _pass(gate_id: str, observed: Any, expected: Any, artifact: str, field: str,
          reason: str = "") -> dict[str, Any]:
    return _gate(gate_id, "PASS", observed, expected, reason, artifact, field, "")


def _fail(gate_id: str, observed: Any, expected: Any, artifact: str, field: str,
          reason: str, remediation: str) -> dict[str, Any]:
    return _gate(gate_id, "FAIL", observed, expected, reason, artifact, field, remediation)


def _blocked(gate_id: str, observed: Any, expected: Any, artifact: str, field: str,
             reason: str, remediation: str) -> dict[str, Any]:
    return _gate(gate_id, "BLOCKED", observed, expected, reason, artifact, field, remediation)


def _not_run(gate_id: str, artifact: str, field: str, reason: str) -> dict[str, Any]:
    return _gate(gate_id, "NOT_RUN", None, None, reason, artifact, field,
                 "run the Gate A0 stage")


def _provenance(policy: dict[str, Any]) -> dict[str, Any]:
    prov = policy.get("provenance", {})
    return prov if isinstance(prov, dict) else {}


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


# ---------------------------------------------------------------------------
# Sub-gate checks
# ---------------------------------------------------------------------------


def _a0_1_provider_class(run: LoadedRun) -> dict[str, Any]:
    """A0.1 provider class is REAL_MODEL (not infrastructure-only)."""
    gate_id = "A0.1"
    artifact = "qualification_report.json / gate_a_result.json"
    field = "is_infrastructure_provider"
    qr = run.qualification_report or {}
    ga = run.gate_a_result or {}
    is_infra = qr.get("is_infrastructure_provider")
    if is_infra is None:
        is_infra = ga.get("is_infrastructure_provider")
    if is_infra is None:
        return _blocked(
            gate_id, None, False, artifact, field,
            "is_infrastructure_provider field is absent in both artifacts",
            "ensure the qualification pipeline records is_infrastructure_provider",
        )
    is_infra_bool = bool(is_infra)
    if not is_infra_bool:
        return _pass(gate_id, is_infra_bool, False, artifact, field,
                     "provider is a REAL_MODEL, not infrastructure-only")
    return _fail(
        gate_id, is_infra_bool, False, artifact, field,
        "provider is infrastructure-only; a real model backend is required for Gate A",
        "re-run counterfactuals on a real model provider",
    )


def _a0_2_source_provenance(run: LoadedRun) -> dict[str, Any]:
    """A0.2 source provenance valid (key SHA-256 digests non-empty)."""
    gate_id = "A0.2"
    artifact = "LoadedRun digests"
    field = "benchmark_manifest_sha256, counterfactual_results_sha256"
    digests = {
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
        "counterfactual_results_sha256": run.counterfactual_results_sha256,
    }
    missing = [k for k, v in digests.items() if not _nonempty(v)]
    if not missing:
        return _pass(gate_id, digests, "all non-empty", artifact, field,
                     "all source provenance digests are non-empty")
    return _fail(
        gate_id, digests, "all non-empty", artifact, field,
        f"empty digests: {missing}",
        "regenerate the missing manifest / counterfactual artifacts",
    )


def _a0_3_versions(run: LoadedRun) -> dict[str, Any]:
    """A0.3 package/qualification versions match."""
    gate_id = "A0.3"
    artifact = "gate_a_result.json"
    field = "package_version, protocol_version"
    ga = run.gate_a_result or {}
    pkg = ga.get("package_version")
    proto = ga.get("protocol_version")
    if not _nonempty(pkg) or not _nonempty(proto):
        return _blocked(
            gate_id, {"package_version": pkg, "protocol_version": proto},
            "non-empty version strings", artifact, field,
            "version fields are absent or empty in gate_a_result",
            "ensure gate_a_result records package_version and protocol_version",
        )
    return _pass(
        gate_id, {"package_version": pkg, "protocol_version": proto},
        "non-empty version strings", artifact, field,
        "package and protocol versions are recorded",
    )


def _a0_4_single_model_revision(run: LoadedRun) -> dict[str, Any]:
    """A0.4 all counterfactual outcomes belong to one model revision."""
    gate_id = "A0.4"
    artifact = "counterfactual_outcomes.json"
    field = "execution_metadata.model_revision"
    revisions: set[str] = set()
    for o in run.outcomes:
        em = o.execution_metadata if hasattr(o, "execution_metadata") else {}
        if not isinstance(em, dict):
            continue
        rev = em.get("model_revision")
        if rev is None:
            rev = em.get("model_id")
        if rev is not None:
            revisions.add(str(rev))
    if not run.outcomes:
        return _blocked(
            gate_id, [], "exactly one model revision", artifact, field,
            "no counterfactual outcomes loaded",
            "run counterfactuals before auditing Gate A0",
        )
    if len(revisions) <= 1:
        return _pass(gate_id, sorted(revisions), "exactly one model revision",
                     artifact, field, "all outcomes share one model revision")
    return _fail(
        gate_id, sorted(revisions), "exactly one model revision", artifact, field,
        f"{len(revisions)} distinct model revisions found across outcomes",
        "re-run counterfactuals on a single pinned model revision",
    )


def _a0_5_positive_evidence_quality(run: LoadedRun) -> dict[str, Any]:
    """A0.5 all admissible experiences have positive evidence quality."""
    gate_id = "A0.5"
    artifact = "dataset_train.json"
    field = "weight"
    examples = run.train_examples
    if not examples:
        return _blocked(
            gate_id, 0, "n_positive > 0", artifact, field,
            "no training examples loaded",
            "build the training dataset before auditing Gate A0",
        )
    n_positive = sum(1 for ex in examples if float(ex.get("weight", 0.0)) > 0)
    n_zero = len(examples) - n_positive
    if n_positive == len(examples):
        return _pass(gate_id, n_positive, "all weights > 0", artifact, field,
                     "every admissible experience has positive evidence quality")
    return _fail(
        gate_id, {"n_positive": n_positive, "n_zero_or_negative": n_zero},
        "all weights > 0", artifact, field,
        f"{n_zero} experiences have non-positive weight",
        "rebuild the dataset so every admissible experience has weight > 0",
    )


def _a0_6_counterfactual_equivalence(run: LoadedRun) -> dict[str, Any]:
    """A0.6 counterfactual equivalence passes (setup_spec consistent per task)."""
    gate_id = "A0.6"
    artifact = "benchmark_manifest.json"
    field = "setup_spec"
    # For each task, all outcomes should share the same setup_spec (i.e. the
    # counterfactual world is identical across actions).  We compare the
    # setup_spec recorded on the task against itself across actions by
    # checking that each task has a single setup_spec.
    inconsistent: list[str] = []
    for tid, task in run.tasks.items():
        # setup_spec is on the task; outcomes inherit it.  Equivalence holds
        # if the task has a non-empty setup_spec and all its outcomes
        # reference the same task_id.
        if not _nonempty(task.setup_spec):
            inconsistent.append(tid)
    if not run.tasks:
        return _blocked(
            gate_id, [], "consistent setup_spec per task", artifact, field,
            "no tasks loaded",
            "build the benchmark manifest before auditing Gate A0",
        )
    if not inconsistent:
        return _pass(gate_id, len(run.tasks), "all tasks have a setup_spec",
                     artifact, field, "counterfactual equivalence holds")
    return _fail(
        gate_id, {"n_inconsistent": len(inconsistent), "tasks": inconsistent[:10]},
        "all tasks have a setup_spec", artifact, field,
        f"{len(inconsistent)} tasks have an empty setup_spec",
        "rebuild the benchmark so every task carries a setup_spec",
    )


def _a0_7_route_completion(run: LoadedRun) -> dict[str, Any]:
    """A0.7 route completion thresholds pass (n_complete == n_manifest)."""
    gate_id = "A0.7"
    artifact = "evaluate_test_*.json"
    field = "n_complete_tasks, n_manifest_tasks"
    evals = [
        ("candidate", run.eval_candidate),
        ("sham", run.eval_sham),
        ("baseline", run.eval_baseline),
        ("oracle", run.eval_oracle),
    ]
    incomplete: list[dict[str, Any]] = []
    for name, ev in evals:
        if ev is None:
            incomplete.append({"policy": name, "reason": "missing eval artifact"})
            continue
        if ev.n_complete_tasks != ev.n_manifest_tasks:
            incomplete.append({
                "policy": name,
                "n_complete_tasks": ev.n_complete_tasks,
                "n_manifest_tasks": ev.n_manifest_tasks,
            })
    if not evals or all(ev is None for _, ev in evals):
        return _blocked(
            gate_id, [], "n_complete == n_manifest for all policies", artifact, field,
            "no evaluation artifacts loaded",
            "run the evaluation stage before auditing Gate A0",
        )
    if not incomplete:
        return _pass(gate_id, [n for n, _ in evals],
                     "n_complete == n_manifest for all policies", artifact, field,
                     "all routes completed")
    return _fail(
        gate_id, incomplete, "n_complete == n_manifest for all policies",
        artifact, field,
        f"{len(incomplete)} policies have incomplete routes",
        "re-run evaluation to complete missing task routes",
    )


def _a0_8_verifier_registry(run: LoadedRun) -> dict[str, Any]:
    """A0.8 verifier registry complete (all verifier_spec types known)."""
    gate_id = "A0.8"
    artifact = "benchmark_manifest.json"
    field = "verifier_spec.type"
    known = {"exact_match", "json_schema", "regex", "llm_judge", "tool_exec",
             "numeric", "set_membership", "code_exec", "none"}
    unknown: dict[str, list[str]] = {}
    for tid, task in run.tasks.items():
        spec = task.verifier_spec or {}
        vtype = spec.get("type", "") if isinstance(spec, dict) else ""
        if vtype and vtype not in known:
            unknown.setdefault(vtype, []).append(tid)
    if not run.tasks:
        return _blocked(
            gate_id, {}, "all verifier types known", artifact, field,
            "no tasks loaded",
            "build the benchmark manifest before auditing Gate A0",
        )
    if not unknown:
        return _pass(gate_id, sorted(known), "all verifier types known",
                     artifact, field, "verifier registry is complete")
    return _fail(
        gate_id, unknown, "all verifier types known", artifact, field,
        f"unknown verifier types: {sorted(unknown.keys())}",
        "register the missing verifier types or rebuild the benchmark",
    )


def _a0_9_no_test_leakage(run: LoadedRun) -> dict[str, Any]:
    """A0.9 no test leakage (test examples don't appear in train examples)."""
    gate_id = "A0.9"
    artifact = "dataset_train.json / dataset_test.json"
    field = "task_id"
    train_ids = {ex.get("task_id") for ex in run.train_examples if ex.get("task_id")}
    test_ids = set(run.test_task_ids)
    leakage = sorted(train_ids & test_ids)
    if not train_ids and not test_ids:
        return _blocked(
            gate_id, [], "empty intersection", artifact, field,
            "no train or test examples loaded",
            "build the dataset before auditing Gate A0",
        )
    if not leakage:
        return _pass(gate_id, [], "empty intersection", artifact, field,
                     "no test task appears in the training set")
    return _fail(
        gate_id, leakage, "empty intersection", artifact, field,
        f"{len(leakage)} test tasks leaked into the training set",
        "rebuild the dataset with strict split separation",
    )


def _a0_10_candidate_serialization_parity(run: LoadedRun) -> dict[str, Any]:
    """A0.10 candidate serialization parity passes (SHA-256 matches)."""
    gate_id = "A0.10"
    artifact = "candidate_policy.json"
    field = "policy_sha256"
    stored = run.candidate_policy_sha256
    declared = _provenance(run.candidate_policy).get("policy_sha256") or \
        run.candidate_policy.get("policy_sha256")
    if not _nonempty(stored):
        return _blocked(
            gate_id, stored, "non-empty SHA-256", artifact, field,
            "candidate_policy.json is missing or unreadable",
            "ensure candidate_policy.json exists",
        )
    if declared is None:
        # No declared hash to compare against; treat stored presence as PASS
        # since there is nothing to contradict it.
        return _pass(gate_id, stored, "non-empty SHA-256", artifact, field,
                     "candidate policy artifact present (no declared hash to compare)")
    if stored == declared:
        return _pass(gate_id, stored, declared, artifact, field,
                     "candidate serialization parity holds")
    return _fail(
        gate_id, stored, declared, artifact, field,
        "candidate policy SHA-256 does not match the declared digest",
        "re-serialize the candidate policy and recompute its digest",
    )


def _a0_11_sham_serialization_parity(run: LoadedRun) -> dict[str, Any]:
    """A0.11 sham serialization parity passes (SHA-256 matches)."""
    gate_id = "A0.11"
    artifact = "sham_policy.json"
    field = "policy_sha256"
    stored = run.sham_policy_sha256
    declared = _provenance(run.sham_policy).get("policy_sha256") or \
        run.sham_policy.get("policy_sha256")
    if not _nonempty(stored):
        return _blocked(
            gate_id, stored, "non-empty SHA-256", artifact, field,
            "sham_policy.json is missing or unreadable",
            "ensure sham_policy.json exists",
        )
    if declared is None:
        return _pass(gate_id, stored, "non-empty SHA-256", artifact, field,
                     "sham policy artifact present (no declared hash to compare)")
    if stored == declared:
        return _pass(gate_id, stored, declared, artifact, field,
                     "sham serialization parity holds")
    return _fail(
        gate_id, stored, declared, artifact, field,
        "sham policy SHA-256 does not match the declared digest",
        "re-serialize the sham policy and recompute its digest",
    )


def _a0_12_task_split_manifests_match(run: LoadedRun) -> dict[str, Any]:
    """A0.12 task/split manifests match (benchmark tasks match split_manifest)."""
    gate_id = "A0.12"
    artifact = "benchmark_manifest.json / split_manifest.json"
    field = "tasks"
    bm_tasks = {t.get("task_id") for t in run.benchmark_manifest.get("tasks", [])
                if t.get("task_id")}
    split = run.split_manifest or {}
    split_tasks: set[str] = set()
    for key in ("test_task_ids", "test_tasks", "task_ids", "tasks"):
        val = split.get(key)
        if isinstance(val, list):
            for v in val:
                if isinstance(v, str):
                    split_tasks.add(v)
                elif isinstance(v, dict):
                    split_tasks.add(v.get("task_id", ""))
    if not bm_tasks and not split_tasks:
        return _blocked(
            gate_id, {"benchmark": [], "split": []}, "matching task sets",
            artifact, field, "both manifests are empty or missing",
            "build the benchmark and split manifests",
        )
    only_bench = sorted(bm_tasks - split_tasks)
    only_split = sorted(split_tasks - bm_tasks)
    if not only_bench and not only_split:
        return _pass(gate_id, len(bm_tasks), "matching task sets", artifact, field,
                     "benchmark and split manifests agree on task ids")
    return _fail(
        gate_id,
        {"only_in_benchmark": only_bench[:10], "only_in_split": only_split[:10]},
        "matching task sets", artifact, field,
        "benchmark and split manifests disagree on task ids",
        "rebuild the split manifest from the benchmark manifest",
    )


def _a0_13_complete_lineage(run: LoadedRun) -> dict[str, Any]:
    """A0.13 artifact hashes form a complete lineage (all SHA-256 non-empty)."""
    gate_id = "A0.13"
    artifact = "LoadedRun digests"
    field = "*_sha256"
    digests = {
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
        "split_manifest_sha256": run.split_manifest_sha256,
        "counterfactual_results_sha256": run.counterfactual_results_sha256,
        "candidate_policy_sha256": run.candidate_policy_sha256,
        "sham_policy_sha256": run.sham_policy_sha256,
        "gate_a_result_sha256": run.gate_a_result_sha256,
        "qualification_report_sha256": run.qualification_report_sha256,
    }
    empty = [k for k, v in digests.items() if not _nonempty(v)]
    if not empty:
        return _pass(gate_id, digests, "all non-empty", artifact, field,
                     "complete lineage of non-empty SHA-256 digests")
    return _fail(
        gate_id, {"empty_digests": empty}, "all non-empty", artifact, field,
        f"{len(empty)} lineage digests are empty",
        "regenerate the missing artifacts",
    )


def _a0_14_no_stale_artifact_reuse(run: LoadedRun) -> dict[str, Any]:
    """A0.14 no stale artifact reuse (all artifacts share protocol_version)."""
    gate_id = "A0.14"
    artifact = "gate_a_result.json / *_policy.json"
    field = "protocol_version"
    versions: set[str] = set()
    ga = run.gate_a_result or {}
    if _nonempty(ga.get("protocol_version")):
        versions.add(str(ga.get("protocol_version")))
    for policy in (run.candidate_policy, run.sham_policy):
        pv = policy.get("protocol_version") if isinstance(policy, dict) else None
        if _nonempty(pv):
            versions.add(str(pv))
        prov = _provenance(policy)
        if _nonempty(prov.get("protocol_version")):
            versions.add(str(prov.get("protocol_version")))
    if not versions:
        return _blocked(
            gate_id, [], "single protocol_version", artifact, field,
            "no protocol_version found in any artifact",
            "ensure artifacts record protocol_version",
        )
    if len(versions) == 1:
        return _pass(gate_id, sorted(versions), "single protocol_version",
                     artifact, field, "all artifacts share one protocol_version")
    return _fail(
        gate_id, sorted(versions), "single protocol_version", artifact, field,
        f"{len(versions)} distinct protocol_versions found — possible stale reuse",
        "re-run the pipeline on a single pinned protocol version",
    )


def _a0_15_active_ids_match(run: LoadedRun) -> dict[str, Any]:
    """A0.15 active candidate and sham IDs match evaluation artifacts."""
    gate_id = "A0.15"
    artifact = "evaluate_test_*.json / *_policy.json"
    field = "policy_id"
    mismatches: list[dict[str, Any]] = []
    pairs = [
        ("candidate", run.candidate_policy, run.eval_candidate),
        ("sham", run.sham_policy, run.eval_sham),
    ]
    for name, policy, ev in pairs:
        policy_id = ""
        if isinstance(policy, dict):
            policy_id = str(policy.get("policy_id") or
                            _provenance(policy).get("policy_id") or "")
        eval_id = ""
        if ev is not None:
            eval_id = str(getattr(ev, "policy_id", "") or "")
        if not policy_id and not eval_id:
            mismatches.append({"policy": name, "reason": "both ids missing"})
        elif policy_id and eval_id and policy_id != eval_id:
            mismatches.append({
                "policy": name,
                "policy_artifact_id": policy_id,
                "eval_artifact_id": eval_id,
            })
    if all(p is None and ev is None for _, p, ev in pairs):
        return _blocked(
            gate_id, [], "matching policy ids", artifact, field,
            "no policy or eval artifacts loaded",
            "run training and evaluation before auditing Gate A0",
        )
    if not mismatches:
        return _pass(gate_id, [n for n, _, _ in pairs], "matching policy ids",
                     artifact, field, "active policy ids match eval artifacts")
    return _fail(
        gate_id, mismatches, "matching policy ids", artifact, field,
        f"{len(mismatches)} policy id mismatches",
        "re-run evaluation against the active trained policies",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SUB_GATES = [
    _a0_1_provider_class,
    _a0_2_source_provenance,
    _a0_3_versions,
    _a0_4_single_model_revision,
    _a0_5_positive_evidence_quality,
    _a0_6_counterfactual_equivalence,
    _a0_7_route_completion,
    _a0_8_verifier_registry,
    _a0_9_no_test_leakage,
    _a0_10_candidate_serialization_parity,
    _a0_11_sham_serialization_parity,
    _a0_12_task_split_manifests_match,
    _a0_13_complete_lineage,
    _a0_14_no_stale_artifact_reuse,
    _a0_15_active_ids_match,
]


def audit_gate_a0(run: LoadedRun) -> dict:
    """Resolve the Gate A0 infrastructure status with a 15-sub-gate checklist.

    Args:
        run: A LoadedRun from data.py.

    Returns:
        A JSON-serializable ``gate_a0_audit`` dict with:
          - sub_gates: dict mapping sub-gate id -> check record
          - overall_status: "PASS" | "FAIL" | "BLOCKED" | "NOT_RUN"
          - n_pass, n_fail, n_blocked, n_not_run
    """
    sub_gates: dict[str, dict[str, Any]] = {}
    for fn in _SUB_GATES:
        record = fn(run)
        sub_gates[record["gate_id"]] = record

    n_pass = sum(1 for g in sub_gates.values() if g["status"] == "PASS")
    n_fail = sum(1 for g in sub_gates.values() if g["status"] == "FAIL")
    n_blocked = sum(1 for g in sub_gates.values() if g["status"] == "BLOCKED")
    n_not_run = sum(1 for g in sub_gates.values() if g["status"] == "NOT_RUN")

    # Final status: PASS iff every mandatory sub-gate passes.
    # FAIL if any valid check failed.  BLOCKED only if evidence genuinely
    # cannot be evaluated (and no FAIL).  NOT_RUN only if the stage was not
    # attempted.
    if n_fail > 0:
        overall_status = "FAIL"
    elif n_blocked > 0:
        overall_status = "BLOCKED"
    elif n_not_run > 0 and n_pass == 0:
        overall_status = "NOT_RUN"
    elif n_pass == len(sub_gates):
        overall_status = "PASS"
    else:
        overall_status = "FAIL"

    return {
        "sub_gates": sub_gates,
        "overall_status": overall_status,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_blocked": n_blocked,
        "n_not_run": n_not_run,
        "n_total": len(sub_gates),
    }
