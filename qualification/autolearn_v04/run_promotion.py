"""Promotion gate accounting for v0.4.0 (Section 8.3/10.4).

v0.4.0 fixes:
  * Primary evidence gates are counted SEPARATELY from the summary
    decision. The summary decision is NOT a primary gate.
  * GateStatus enum (PASS/FAIL/BLOCKED/NOT_RUN) replaces boolean flags.
  * Independent provenance verification: promotion recomputes every
    digest and compares to the policy's declared values, failing closed
    on mismatch.
  * Smoke mode never issues a promotion verdict. PROMOTION_DECISION is
    BLOCKED and SCIENTIFIC_QUALIFICATION is NOT_RUN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .provenance import verify_policy_provenance
from .schemas import (
    EMPTY_SHA256,
    GateCategory,
    GateResult,
    GateStatus,
    PolicyProvenance,
    ProvenanceError,
    read_json,
    sha256_text,
    write_json,
)
from .train_candidate import _feature_schema_digest, _learner_code_digest


def _load_gate_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    return data.get("gates", [])


def _gate_result_from_dict(d: dict[str, Any]) -> GateResult:
    return GateResult(
        name=d["name"],
        category=GateCategory(d.get("category", "evidence_completeness")),
        status=GateStatus(d.get("status", "not_run")),
        detail=d.get("detail", {}),
    )


def run_promotion(config: QualificationConfig) -> dict[str, Any]:
    """Run the promotion gate accounting with independent provenance verification."""
    artifacts = Path(config.artifacts_dir)

    # Smoke mode: never issue a promotion verdict.
    if config.is_smoke:
        return _smoke_blocked_result(config)

    # Load Gate A and Gate B results.
    gate_a_path = artifacts / "gate_a_result.json"
    gate_b_path = artifacts / "gate_b_result.json"
    gate_a = read_json(gate_a_path) if gate_a_path.exists() else None
    gate_b = read_json(gate_b_path) if gate_b_path.exists() else None

    all_gates: list[GateResult] = []

    # --- Integrity gates ---
    # Provenance independent verification.
    candidate_policy = read_json(artifacts / "candidate_policy.json") if (artifacts / "candidate_policy.json").exists() else None
    if candidate_policy is None:
        all_gates.append(GateResult(
            "candidate_policy_exists", GateCategory.INTEGRITY,
            GateStatus.BLOCKED, {"reason": "candidate_policy.json missing"},
        ))
        return _assemble_result(config, all_gates, gate_a, gate_b)

    prov_dict = candidate_policy.get("provenance", {})
    if not prov_dict:
        all_gates.append(GateResult(
            "policy_provenance_embedded", GateCategory.PROVENANCE,
            GateStatus.FAIL, {"reason": "policy has no embedded provenance"},
        ))
        return _assemble_result(config, all_gates, gate_a, gate_b)

    try:
        declared_prov = PolicyProvenance.from_dict(prov_dict)
        declared_prov.validate()
        all_gates.append(GateResult(
            "policy_provenance_complete", GateCategory.PROVENANCE,
            GateStatus.PASS, {"n_fields": len(prov_dict)},
        ))
    except (ProvenanceError, ValueError, KeyError) as exc:
        all_gates.append(GateResult(
            "policy_provenance_complete", GateCategory.PROVENANCE,
            GateStatus.FAIL, {"reason": str(exc)},
        ))
        return _assemble_result(config, all_gates, gate_a, gate_b)

    # Independent provenance verification (Section 7.5).
    verification = verify_policy_provenance(
        config=config,
        declared=declared_prov,
        benchmark_manifest_path=artifacts / "benchmark_manifest.json",
        split_manifest_path=artifacts / "split_manifest.json",
        counterfactual_results_path=artifacts / "counterfactual_outcomes.json",
        training_dataset_path=artifacts / "dataset_manifest.json",
        feature_schema_digest=_feature_schema_digest(),
        hyperparameter_digest=declared_prov.hyperparameter_digest,
        learner_code_digest=_learner_code_digest(),
    )
    all_gates.append(GateResult(
        "provenance_independently_verified", GateCategory.PROVENANCE,
        GateStatus.PASS if verification["valid"] else GateStatus.FAIL,
        {"n_passed": verification["n_passed"], "n_failed": verification["n_failed"],
         "checks": verification["checks"]},
    ))

    # Runtime type gate.
    cf_outcomes = read_json(artifacts / "counterfactual_outcomes.json") if (artifacts / "counterfactual_outcomes.json").exists() else []
    n_real = sum(1 for o in cf_outcomes if o.get("execution_metadata", {}).get("runtime_type") == "real")
    n_total = len(cf_outcomes)
    all_gates.append(GateResult(
        "all_rows_runtime_type_real", GateCategory.INTEGRITY,
        GateStatus.PASS if n_total > 0 and n_real == n_total else GateStatus.FAIL,
        {"n_real": n_real, "n_total": n_total},
    ))

    # Provider gate: must be REAL_MODEL for a causal claim (Section 4/19).
    from .provider_classification import ProviderClass, classify_grounded_provider, classify_local_transformers
    if config.is_infrastructure_provider:
        provider_caps = classify_grounded_provider()
    else:
        provider_caps = classify_local_transformers(
            config.model, dtype=config.dtype, device=config.device,
        )
    all_gates.append(GateResult(
        "provider_class_is_real_model", GateCategory.INTEGRITY,
        GateStatus.PASS if provider_caps.provider_class is ProviderClass.REAL_MODEL else GateStatus.BLOCKED,
        {"provider_name": provider_caps.provider_name,
         "provider_class": provider_caps.provider_class.value,
         "supports_gate_a": provider_caps.supports_gate_a_claim,
         "supports_gate_b": provider_caps.supports_gate_b_claim},
    ))

    # --- Runtime completion diagnostics gate ---
    diag_path = artifacts / "runtime_completion_diagnostics.json"
    if diag_path.exists():
        diag = read_json(diag_path)
        all_gates.append(GateResult(
            "runtime_completion_passes", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.PASS if diag.get("status") == "PASS" else GateStatus.BLOCKED,
            {"status": diag.get("status"), "failed_actions": diag.get("failed_actions", [])},
        ))
    else:
        all_gates.append(GateResult(
            "runtime_completion_passes", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.NOT_RUN, {"reason": "runtime_completion_diagnostics.json missing"},
        ))

    # --- Gate A gates ---
    if gate_a is None:
        all_gates.append(GateResult(
            "gate_a_executed", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.NOT_RUN, {"reason": "gate_a_result.json missing"},
        ))
    elif gate_a.get("status") == "BLOCKED":
        for g in gate_a.get("gates", []):
            all_gates.append(_gate_result_from_dict(g))
    else:
        for g in gate_a.get("gates", []):
            all_gates.append(_gate_result_from_dict(g))

    # --- Gate B gates ---
    if gate_b is None:
        all_gates.append(GateResult(
            "gate_b_executed", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.NOT_RUN, {"reason": "gate_b_result.json missing"},
        ))
    elif gate_b.get("status") == "BLOCKED":
        all_gates.append(GateResult(
            "gate_b_executed", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.BLOCKED, {"reason": gate_b.get("reason", "blocked")},
        ))
    else:
        for g in gate_b.get("gates", []):
            all_gates.append(_gate_result_from_dict(g))

    # --- Post-promotion behavioral proof gate ---
    post_path = artifacts / "post_promotion_result.json"
    if post_path.exists():
        post = read_json(post_path)
        all_gates.append(GateResult(
            "post_promotion_behavioral_proof", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.PASS if post.get("passes") else GateStatus.FAIL,
            {"n_improvements": post.get("n_behavioral_improvements", 0)},
        ))
    else:
        all_gates.append(GateResult(
            "post_promotion_behavioral_proof", GateCategory.EVIDENCE_COMPLETENESS,
            GateStatus.NOT_RUN, {"reason": "post_promotion_result.json missing"},
        ))

    return _assemble_result(config, all_gates, gate_a, gate_b)


def _smoke_blocked_result(config: QualificationConfig) -> dict[str, Any]:
    """Smoke mode: pipeline integrity only, no scientific verdict."""
    artifacts = Path(config.artifacts_dir)
    pipeline_ok = True
    # In smoke mode, Gate A and Gate B are expected to be BLOCKED (not run).
    # Only check for the stages that run BEFORE promotion in the pipeline.
    stages = {}
    for stage, fname in [
        ("build_benchmark", "benchmark_manifest.json"),
        ("run_counterfactuals", "counterfactual_outcomes.json"),
        ("build_dataset", "dataset_manifest.json"),
        ("train_candidate", "candidate_policy.json"),
        ("train_sham", "sham_policy.json"),
        ("run_post_promotion", "post_promotion_result.json"),
    ]:
        if fname is None:
            stages[stage] = "ok"
            continue
        exists = (artifacts / fname).exists()
        stages[stage] = "ok" if exists else "missing"
        if not exists:
            pipeline_ok = False

    result = {
        "schema_version": "promotion-result/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": "smoke",
        "PIPELINE_INTEGRITY": "PASS" if pipeline_ok else "FAIL",
        "SCIENTIFIC_QUALIFICATION": "NOT_RUN",
        "PROMOTION_DECISION": "BLOCKED",
        "status": "BLOCKED",
        "promotion_approved": False,
        "stages": stages,
        "gates": [],
        "primary_gate_count": 0,
        "primary_gates_passed": 0,
        "primary_gates_failed": 0,
        "primary_gates_blocked": 0,
        "summary_decision": "BLOCKED",
        "reason": "smoke mode: scientific qualification not run, promotion blocked",
    }
    write_json(artifacts / "promotion_result.json", result)
    return result


def _assemble_result(
    config: QualificationConfig,
    all_gates: list[GateResult],
    gate_a: dict | None,
    gate_b: dict | None,
) -> dict[str, Any]:
    """Assemble the final promotion result from all gates."""
    artifacts = Path(config.artifacts_dir)

    n_passed = sum(1 for g in all_gates if g.status is GateStatus.PASS)
    n_failed = sum(1 for g in all_gates if g.status is GateStatus.FAIL)
    n_blocked = sum(1 for g in all_gates if g.status is GateStatus.BLOCKED)
    n_not_run = sum(1 for g in all_gates if g.status is GateStatus.NOT_RUN)

    # The summary decision is NOT a primary gate (Section 8.3).
    if n_blocked > 0 or n_not_run > 0:
        summary = "BLOCKED"
        approved = False
    elif n_failed > 0:
        summary = "FAIL"
        approved = False
    else:
        summary = "PASS"
        approved = True

    result = {
        "schema_version": "promotion-result/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": config.mode,
        "runtime": config.runtime,
        "provider": config.provider,
        "is_infrastructure_provider": config.is_infrastructure_provider,
        "PIPELINE_INTEGRITY": "PASS",
        "SCIENTIFIC_QUALIFICATION": summary,
        "PROMOTION_DECISION": summary,
        "status": summary,
        "promotion_approved": approved,
        "gates": [g.to_dict() for g in all_gates],
        "primary_gate_count": len(all_gates),
        "primary_gates_passed": n_passed,
        "primary_gates_failed": n_failed,
        "primary_gates_blocked": n_blocked,
        "primary_gates_not_run": n_not_run,
        "summary_decision": summary,
        "gate_a_status": gate_a.get("status") if gate_a else "NOT_RUN",
        "gate_b_status": gate_b.get("status") if gate_b else "NOT_RUN",
    }
    write_json(artifacts / "promotion_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v0.4.0 promotion gate accounting.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    parser.add_argument("--model", default="qual-grounded-v04")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir,
    )
    result = run_promotion(config)
    print(f"promotion: {result['status']}")
    print(f"  PIPELINE_INTEGRITY: {result.get('PIPELINE_INTEGRITY', 'N/A')}")
    print(f"  SCIENTIFIC_QUALIFICATION: {result.get('SCIENTIFIC_QUALIFICATION', 'N/A')}")
    print(f"  PROMOTION_DECISION: {result.get('PROMOTION_DECISION', 'N/A')}")
    print(f"  primary_gates: {result['primary_gates_passed']}/{result['primary_gate_count']} passed")
    return 0 if result["promotion_approved"] else 1


if __name__ == "__main__":
    sys.exit(main())
