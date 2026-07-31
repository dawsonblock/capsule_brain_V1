"""Main orchestrator for v0.4.3 repair analysis.

Runs all repair modules against the frozen 7B run artifacts and produces
all required v0.4.3 artifacts. No new model executions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .data import load_run, LoadedRun
from .config import (
    PACKAGE_VERSION, QUALIFICATION_VERSION, AUTOLEARN_VERSION,
    DEFAULT_N_BOOTSTRAP, DEFAULT_SEED, SHAM_N_SEEDS, SHAM_BASE_SEED,
    CANDIDATE_N_SEEDS, CANDIDATE_BASE_SEED,
)


def run_all_v043_diagnostics(
    artifacts_dir: str | Path,
    output_dir: str | Path,
    run_id: str = "7b_full_001",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run all v0.4.3 repair diagnostics and write all required artifacts.

    Args:
        artifacts_dir: path to the frozen 7B run artifacts.
        output_dir: directory to write diagnostic artifacts.
        run_id: identifier for the run.
        repo_root: repository root for git SHA.

    Returns:
        analysis_manifest dict.
    """
    start_time = time.time()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    print("=" * 70)
    print("AutoLearn v0.4.3 — Gate A Decision Repair Analysis")
    print("=" * 70)
    print(f"Input artifacts: {artifacts_dir}")
    print(f"Output directory: {output}")
    print(f"Run ID: {run_id}")
    print()

    commit_sha = _get_commit_sha(repo_root)
    artifacts_produced: dict[str, str] = {}

    # 1. Load the frozen run.
    print("[1/24] Loading frozen run artifacts...")
    run = load_run(artifacts_dir)
    print(f"  Loaded {len(run.tasks)} tasks, {len(run.outcomes)} outcomes")
    print(f"  Test tasks: {len(run.test_task_ids)}")
    print(f"  Candidate mean utility (artifact): {run.eval_candidate.mean_utility:.6f}")
    print(f"  Sham mean utility (artifact):      {run.eval_sham.mean_utility:.6f}")
    print(f"  Baseline mean utility (artifact):   {run.eval_baseline.mean_utility:.6f}")
    print(f"  Oracle mean utility (artifact):     {run.eval_oracle.mean_utility:.6f}")

    # 2. Canonical policy evaluation.
    print("\n[2/24] Running canonical policy evaluation...")
    from .canonical_evaluator import evaluate_all_policies, result_to_dict
    canonical_results = evaluate_all_policies(run, seed=DEFAULT_SEED)
    canonical_output = {name: result_to_dict(res) for name, res in canonical_results.items()}
    _write_json(output / "canonical_policy_evaluation.json", canonical_output)
    artifacts_produced["canonical_policy_evaluation.json"] = _sha256_file(output / "canonical_policy_evaluation.json")
    for name, res in canonical_results.items():
        print(f"  {name:25s}: mean_utility={res.mean_utility:.6f}")

    # 3. Metric consistency.
    print("\n[3/24] Checking metric consistency...")
    from .metric_consistency import check_metric_consistency
    mc_result = check_metric_consistency(run, canonical_results)
    _write_json(output / "metric_consistency_report.json", mc_result)
    artifacts_produced["metric_consistency_report.json"] = _sha256_file(output / "metric_consistency_report.json")
    print(f"  Overall: {mc_result['overall_status']}")
    for check in mc_result.get("failed_checks", []):
        if isinstance(check, dict):
            print(f"  FAILED: {check.get('name', '')} (expected={check.get('expected')}, actual={check.get('actual')})")
        else:
            print(f"  FAILED: {check}")

    # 4. Sham identity.
    print("\n[4/24] Validating sham identity...")
    from .sham_identity import validate_sham_identity
    sham_id = validate_sham_identity(run)
    _write_json(output / "sham_identity_report.json", sham_id)
    artifacts_produced["sham_identity_report.json"] = _sha256_file(output / "sham_identity_report.json")
    print(f"  Overall: {sham_id.get('overall_status', 'UNKNOWN')}")

    # 5. Gate A0 audit.
    print("\n[5/24] Auditing Gate A0 infrastructure...")
    from .gate_a0_audit import audit_gate_a0
    a0_result = audit_gate_a0(run)
    _write_json(output / "gate_a0_audit.json", a0_result)
    artifacts_produced["gate_a0_audit.json"] = _sha256_file(output / "gate_a0_audit.json")
    print(f"  Overall: {a0_result['overall_status']}")
    print(f"  Pass: {a0_result['n_pass']}, Fail: {a0_result['n_fail']}, Blocked: {a0_result['n_blocked']}")

    # 6. Decision trace.
    print("\n[6/24] Reconstructing candidate decision trace...")
    from .decision_trace import reconstruct_decision_trace
    dt_result = reconstruct_decision_trace(run)
    _write_jsonl(output / "candidate_decision_trace.jsonl", dt_result["per_task"])
    artifacts_produced["candidate_decision_trace.jsonl"] = _sha256_file(output / "candidate_decision_trace.jsonl")
    print(f"  Traced {len(dt_result['per_task'])} tasks")
    for stage, dist in dt_result.get("action_distributions", {}).items():
        print(f"  {stage}: {dist}")

    # 7. Raw vs executed.
    print("\n[7/24] Comparing raw vs executed candidate...")
    from .raw_vs_executed import compare_raw_vs_executed
    rve_result = compare_raw_vs_executed(run, seed=DEFAULT_SEED)
    _write_json(output / "raw_vs_executed_results.json", rve_result)
    artifacts_produced["raw_vs_executed_results.json"] = _sha256_file(output / "raw_vs_executed_results.json")
    print(f"  Raw utility:      {rve_result['raw_result']['mean_utility']:.6f}")
    print(f"  Executed utility: {rve_result['executed_result']['mean_utility']:.6f}")
    print(f"  Bottleneck: {rve_result['bottleneck_classification']}")

    # 8. Calibration fallback audit.
    print("\n[8/24] Auditing calibration and fallback...")
    from .calibration_fallback_audit import audit_calibration_fallback
    cal_result = audit_calibration_fallback(run, seed=DEFAULT_SEED)
    _write_json(output / "calibration_fallback_audit.json", cal_result)
    artifacts_produced["calibration_fallback_audit.json"] = _sha256_file(output / "calibration_fallback_audit.json")
    print(f"  Production threshold: {cal_result.get('production_threshold', 'N/A')}")

    # 9. Headroom concentration.
    print("\n[9/24] Analyzing headroom concentration...")
    from .headroom_concentration import analyze_headroom_concentration
    hc_result = analyze_headroom_concentration(run)
    _write_json(output / "headroom_concentration.json", hc_result)
    artifacts_produced["headroom_concentration.json"] = _sha256_file(output / "headroom_concentration.json")
    print(f"  Concentration: {hc_result.get('concentration_status', 'UNKNOWN')}")
    hhi = hc_result.get("concentration_metrics", {}).get("hhi", 0)
    print(f"  HHI: {hhi:.4f}")

    # 10. High margin recovery.
    print("\n[10/24] Evaluating high-margin recovery...")
    from .high_margin_recovery import evaluate_high_margin_recovery
    hm_result = evaluate_high_margin_recovery(run, seed=DEFAULT_SEED)
    _write_json(output / "high_margin_recovery.json", hm_result)
    artifacts_produced["high_margin_recovery.json"] = _sha256_file(output / "high_margin_recovery.json")
    ga5 = hm_result.get("gate_a5", {})
    print(f"  Gate A5: {ga5.get('status', 'UNKNOWN')}")

    # 11. Feature signal.
    print("\n[11/24] Analyzing feature signal...")
    from .feature_signal import analyze_feature_signal
    fs_result = analyze_feature_signal(run, seed=DEFAULT_SEED)
    _write_json(output / "feature_signal_analysis.json", fs_result)
    artifacts_produced["feature_signal_analysis.json"] = _sha256_file(output / "feature_signal_analysis.json")
    print(f"  Conclusion: {fs_result.get('learnability_conclusion', 'UNKNOWN')}")

    # 12. Missing state analysis.
    print("\n[12/24] Analyzing missing state...")
    from .missing_state_analysis import analyze_missing_state
    ms_result = analyze_missing_state(run)
    _write_json(output / "missing_state_analysis.json", ms_result)
    artifacts_produced["missing_state_analysis.json"] = _sha256_file(output / "missing_state_analysis.json")
    n_candidates = len(ms_result.get("missing_feature_candidates", []))
    print(f"  Missing feature candidates: {n_candidates}")

    # 13. Feature registry.
    print("\n[13/24] Building feature registry...")
    from .feature_registry import build_feature_registry
    fr_result = build_feature_registry(run)
    _write_json(output / "feature_registry.json", fr_result)
    artifacts_produced["feature_registry.json"] = _sha256_file(output / "feature_registry.json")
    print(f"  Accepted: {fr_result.get('n_accepted', 0)}, Rejected: {fr_result.get('n_rejected', 0)}")

    # 14. Learner validation.
    print("\n[14/24] Validating learners...")
    from .learner_validation import validate_learners
    lv_result = validate_learners(run, seed=DEFAULT_SEED)
    _write_json(output / "learner_validation_results.json", lv_result)
    artifacts_produced["learner_validation_results.json"] = _sha256_file(output / "learner_validation_results.json")
    print(f"  Best validation learner: {lv_result.get('best_validation_learner', 'N/A')}")

    # 15. Soft utility targets.
    print("\n[15/24] Comparing soft utility targets...")
    from .soft_utility_targets import compare_soft_targets
    su_result = compare_soft_targets(run, seed=DEFAULT_SEED)
    _write_json(output / "soft_utility_results.json", su_result)
    artifacts_produced["soft_utility_results.json"] = _sha256_file(output / "soft_utility_results.json")
    print(f"  Best tau: {su_result.get('best_tau', 'N/A')}")

    # 16. Class balance.
    print("\n[16/24] Analyzing class balance...")
    from .class_balance import analyze_class_balance
    cb_result = analyze_class_balance(run, seed=DEFAULT_SEED)
    _write_json(output / "class_balance_results.json", cb_result)
    artifacts_produced["class_balance_results.json"] = _sha256_file(output / "class_balance_results.json")
    print(f"  Best option: {cb_result.get('best_option', 'N/A')}")

    # 17. Sham ensemble.
    print("\n[17/24] Running sham ensemble (50 seeds)...")
    from .sham_ensemble import run_sham_ensemble
    se_result = run_sham_ensemble(run, n_seeds=SHAM_N_SEEDS, base_seed=SHAM_BASE_SEED)
    _write_json(output / "sham_ensemble_results.json", se_result)
    artifacts_produced["sham_ensemble_results.json"] = _sha256_file(output / "sham_ensemble_results.json")
    print(f"  Primary sham mean: {se_result.get('primary_sham_mean', 0):.6f}")
    print(f"  Candidate beats sham: {se_result.get('candidate_beats_sham', False)}")

    # 18. Candidate stability.
    print("\n[18/24] Analyzing candidate stability (50 seeds)...")
    from .candidate_stability import analyze_candidate_stability
    cs_result = analyze_candidate_stability(run, n_seeds=CANDIDATE_N_SEEDS, base_seed=CANDIDATE_BASE_SEED)
    _write_json(output / "candidate_stability_results.json", cs_result)
    artifacts_produced["candidate_stability_results.json"] = _sha256_file(output / "candidate_stability_results.json")
    print(f"  Stability: {cs_result.get('stability_classification', 'UNKNOWN')}")

    # 19. Parameter diagnostics.
    print("\n[19/24] Diagnosing router parameters...")
    from .parameter_diagnostics import diagnose_parameters
    pd_result = diagnose_parameters(run, seed=DEFAULT_SEED)
    _write_json(output / "router_parameter_diagnostics.json", pd_result)
    artifacts_produced["router_parameter_diagnostics.json"] = _sha256_file(output / "router_parameter_diagnostics.json")

    # 20. Utility consistency.
    print("\n[20/24] Auditing utility consistency...")
    from .utility_consistency import audit_utility_consistency
    uc_result = audit_utility_consistency(run)
    _write_json(output / "utility_consistency_report.json", uc_result)
    artifacts_produced["utility_consistency_report.json"] = _sha256_file(output / "utility_consistency_report.json")
    print(f"  Overall: {uc_result.get('overall_status', 'UNKNOWN')}")

    # 21. Counterfactual completeness.
    print("\n[21/24] Auditing counterfactual completeness...")
    from .counterfactual_completeness import audit_counterfactual_completeness
    cc_result = audit_counterfactual_completeness(run)
    _write_json(output / "counterfactual_completeness.json", cc_result)
    artifacts_produced["counterfactual_completeness.json"] = _sha256_file(output / "counterfactual_completeness.json")
    summary = cc_result.get("summary", {})
    print(f"  Complete: {summary.get('COMPLETE', 0)}, Incomplete: {summary.get('INCOMPLETE', 0)}")

    # 22. Gate A layered.
    print("\n[22/24] Evaluating corrected Gate A layers...")
    from .gate_a_layers import evaluate_gate_a_v043
    gate_a = evaluate_gate_a_v043(
        run, a0_result, hc_result, fs_result, canonical_results,
        hm_result, se_result, mc_result,
        n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED,
    )
    _write_json(output / "gate_a_v043_results.json", gate_a)
    artifacts_produced["gate_a_v043_results.json"] = _sha256_file(output / "gate_a_v043_results.json")
    print(f"  Overall: {gate_a['overall_status']}")
    for name, gate in gate_a["sub_gates"].items():
        print(f"    {name}: {gate['status']}")

    # 23. Scale decision.
    print("\n[23/24] Making scale decision...")
    from .scale_decision import make_scale_decision_v043
    scale_decision = make_scale_decision_v043(
        run, gate_a, hc_result, fs_result, rve_result,
        cs_result, se_result, cc_result, uc_result, lv_result,
    )
    _write_json(output / "model_scale_decision.json", scale_decision)
    artifacts_produced["model_scale_decision.json"] = _sha256_file(output / "model_scale_decision.json")
    print(f"  Decision: {scale_decision['decision']}")
    print(f"  Reason: {scale_decision['reason']}")

    # 24. Matched pilot manifest + report.
    print("\n[24/24] Generating pilot manifest and report...")
    from .matched_pilot_manifest import check_pilot_eligibility, generate_pilot_manifest
    pilot_eligibility = check_pilot_eligibility(
        gate_a0_status=a0_result["overall_status"],
        counterfactual_completeness_status=cc_result.get("overall_status", "FAIL"),
        utility_consistency_status=uc_result.get("overall_status", "FAIL"),
        sham_validity=sham_id.get("overall_status") == "PASS",
        oracle_headroom=canonical_results["oracle"].mean_utility - canonical_results["sham"].mean_utility,
        headroom_concentration_status=hc_result.get("concentration_status", "CONCENTRATED"),
        feature_signal_conclusion=fs_result.get("learnability_conclusion", "FEATURE_SIGNAL_ABSENT"),
        candidate_stability_classification=cs_result.get("stability_classification", "UNSTABLE"),
        scale_decision=scale_decision["decision"],
    )
    pilot_manifest = generate_pilot_manifest(run, pilot_eligibility)
    if pilot_manifest.get("pilot_approved"):
        _write_json(output / "matched_pilot_manifest.json", pilot_manifest)
        artifacts_produced["matched_pilot_manifest.json"] = _sha256_file(output / "matched_pilot_manifest.json")
        print(f"  Pilot approved: {pilot_manifest['pilot_manifest']['n_tasks']} tasks")
    else:
        print(f"  Pilot not approved: {pilot_manifest.get('reason', 'N/A')}")

    # Generate report.
    from .report_generator import generate_v043_report
    report = generate_v043_report(
        run, canonical_results, mc_result, sham_id, a0_result,
        dt_result, rve_result, cal_result, hc_result, hm_result,
        fs_result, ms_result, fr_result, lv_result, su_result,
        cb_result, se_result, cs_result, pd_result, uc_result,
        cc_result, gate_a, scale_decision, pilot_manifest,
    )
    report_path = output / "GATE_A_V043_REPAIR_REPORT.md"
    report_path.write_text(report)
    artifacts_produced["GATE_A_V043_REPAIR_REPORT.md"] = _sha256_file(report_path)

    # Write analysis manifest.
    elapsed = time.time() - start_time
    config_digest = _sha256_json({
        "run_id": run_id,
        "artifacts_dir": str(artifacts_dir),
        "n_bootstrap": DEFAULT_N_BOOTSTRAP,
        "n_sham_seeds": SHAM_N_SEEDS,
        "n_candidate_seeds": CANDIDATE_N_SEEDS,
        "seed": DEFAULT_SEED,
    })

    manifest = {
        "analysis_run_id": f"{run_id}_v043_repair",
        "original_run_id": run_id,
        "original_commit_sha": commit_sha,
        "original_candidate_sha256": run.candidate_policy_sha256,
        "original_sham_sha256": run.sham_policy_sha256,
        "original_counterfactual_results_sha256": run.counterfactual_results_sha256,
        "original_gate_a_results_sha256": run.gate_a_result_sha256,
        "v043_source_sha256": _sha256_file(Path(__file__)),
        "analysis_config_sha256": config_digest,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "artifacts_produced": artifacts_produced,
        "n_artifacts": len(artifacts_produced),
        "random_seeds": {
            "bootstrap": DEFAULT_SEED,
            "sham_base": SHAM_BASE_SEED,
            "candidate_base": CANDIDATE_BASE_SEED,
        },
        "elapsed_seconds": elapsed,
        "timestamp": _timestamp(),
    }
    _write_json(output / "analysis_manifest.json", manifest)
    artifacts_produced["analysis_manifest.json"] = _sha256_file(output / "analysis_manifest.json")

    print(f"\n{'=' * 70}")
    print(f"v0.4.3 Repair Analysis Complete — {len(artifacts_produced)} artifacts produced")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output}")
    print(f"{'=' * 70}")

    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _get_commit_sha(repo_root: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
