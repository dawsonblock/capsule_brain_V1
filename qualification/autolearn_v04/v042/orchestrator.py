"""Main orchestrator for v0.4.2 diagnostic analysis.

Runs all diagnostic modules against the existing 7B full-run artifacts
and produces all required artifacts. No new model executions.
"""
from __future__ import annotations

import json
import hashlib
import time
import subprocess
from pathlib import Path
from typing import Any

from .data import load_run, LoadedRun
from .archive_run import archive_scientific_run, get_commit_sha


def run_all_diagnostics(
    artifacts_dir: str | Path,
    output_dir: str | Path,
    run_id: str = "7b_full_001",
    repo_root: str | Path | None = None,
    mini_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all v0.4.2 diagnostics and write all required artifacts.

    Args:
        artifacts_dir: path to the 7B full-run artifacts.
        output_dir: directory to write diagnostic artifacts.
        run_id: identifier for the archived run.
        repo_root: repository root for git SHA.
        mini_dir: path to the mini run artifacts for cross-run comparison.

    Returns:
        analysis_manifest dict.
    """
    start_time = time.time()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    print("=" * 70)
    print("AutoLearn v0.4.2 — Routing-Headroom Analysis")
    print("=" * 70)
    print(f"Input artifacts: {artifacts_dir}")
    print(f"Output directory: {output}")
    print(f"Run ID: {run_id}")
    print()

    # 1. Archive the run as immutable.
    print("[1/20] Archiving scientific run as immutable...")
    archive_manifest = archive_scientific_run(
        artifacts_dir=artifacts_dir,
        archive_root=output / "archived_runs",
        run_id=run_id,
        repo_root=repo_root,
    )
    _write_json(output / "archived_runs" / "scientific" / run_id / "archive_manifest.json",
                archive_manifest)
    print(f"  Archived as {run_id} (IMMUTABLE, SCIENTIFIC NEGATIVE RESULT)")

    # Load the run.
    print("\n[2/20] Loading run artifacts...")
    run = load_run(artifacts_dir)
    print(f"  Loaded {len(run.tasks)} tasks, {len(run.outcomes)} outcomes")
    print(f"  Actions: {run.all_actions}")
    print(f"  Families: {run.all_families}")
    print(f"  Splits: {run.all_splits}")

    test_ids = run.test_task_ids
    print(f"  Test tasks: {len(test_ids)}")

    commit_sha = get_commit_sha(repo_root)
    config_digest = _sha256_json({
        "run_id": run_id,
        "artifacts_dir": str(artifacts_dir),
        "epsilon_baseline": 0.01,
        "epsilon_sham": 0.01,
        "n_bootstrap": 10000,
        "n_seeds": 20,
        "seed": 42,
    })

    artifacts_produced = {}

    # 3. Policy control ladder.
    print("\n[3/20] Evaluating policy control ladder...")
    from .policy_controls import evaluate_policy_ladder
    ladder = evaluate_policy_ladder(run, task_ids=test_ids, seed=42)
    _write_json(output / "policy_control_ladder.json", ladder)
    artifacts_produced["policy_control_ladder.json"] = _sha256_file(output / "policy_control_ladder.json")
    print(f"  Evaluated {len(ladder['policies'])} policies")
    for name, res in ladder["policies"].items():
        mu = res.get("mean_utility", 0.0)
        print(f"    {name}: mean_utility={mu:.4f}")

    # 4. Oracle routing headroom.
    print("\n[4/20] Computing oracle routing headroom...")
    from .headroom import compute_headroom
    headroom = compute_headroom(run, task_ids=test_ids, seed=42)
    _write_json(output / "oracle_headroom_analysis.json", headroom)
    artifacts_produced["oracle_headroom_analysis.json"] = _sha256_file(output / "oracle_headroom_analysis.json")
    print(f"  H_oracle_sham={headroom['H_oracle_sham']:.4f}")
    print(f"  R_sham={headroom['R_sham']:.4f}")
    print(f"  Status: {headroom['routing_headroom_status']}")

    # 5. Best-action imbalance.
    print("\n[5/20] Analyzing best-action imbalance...")
    from .imbalance import analyze_imbalance
    imbalance = analyze_imbalance(run)
    _write_json(output / "best_action_distribution.json", imbalance)
    artifacts_produced["best_action_distribution.json"] = _sha256_file(output / "best_action_distribution.json")
    for split_name, split_data in imbalance.get("splits", {}).items():
        ent = split_data.get("class_entropy", 0)
        neff = split_data.get("effective_number_of_actions", 0)
        maj = split_data.get("majority_action", "")
        maj_rate = split_data.get("majority_action_rate", 0)
        print(f"  {split_name}: entropy={ent:.3f}, N_eff={neff:.2f}, majority={maj} ({maj_rate:.1%})")

    # 6. Utility-margin analysis.
    print("\n[6/20] Computing utility-margin analysis...")
    from .margin_analysis import analyze_margins
    margins = analyze_margins(run, task_ids=test_ids, seed=42)
    _write_json(output / "utility_margin_analysis.json", margins)
    artifacts_produced["utility_margin_analysis.json"] = _sha256_file(output / "utility_margin_analysis.json")
    for bucket_name, bucket_data in margins.get("buckets", {}).items():
        n = bucket_data.get("n_tasks", bucket_data.get("task_count", 0))
        cu = bucket_data.get("candidate_utility", 0)
        print(f"  {bucket_name}: n={n}, candidate_utility={cu:.4f}")

    # 7. Sham audit + seed distribution.
    print("\n[7/20] Auditing sham construction...")
    from .sham_audit import audit_sham_construction, compute_sham_seed_distribution
    sham_audit = audit_sham_construction(run)
    _write_json(output / "sham_audit.json", sham_audit)
    artifacts_produced["sham_audit.json"] = _sha256_file(output / "sham_audit.json")

    print("  Running 20 sham seeds...")
    sham_dist = compute_sham_seed_distribution(run, n_seeds=20, task_ids=test_ids)
    _write_json(output / "sham_seed_distribution.json", sham_dist)
    artifacts_produced["sham_seed_distribution.json"] = _sha256_file(output / "sham_seed_distribution.json")
    print(f"  Sham utility: mean={sham_dist['sham_mean']:.4f}, std={sham_dist['sham_std']:.4f}")
    print(f"  Candidate percentile: {sham_dist['candidate_percentile']:.1f}%")

    # 8. Confusion matrices.
    print("\n[8/20] Computing confusion matrices...")
    from .confusion import compute_confusion_analysis
    confusion = compute_confusion_analysis(run, task_ids=test_ids)
    _write_json(output / "candidate_confusion_matrix.json", confusion["candidate_confusion_matrix"])
    _write_json(output / "utility_weighted_confusion_matrix.json", confusion["utility_weighted_confusion_matrix"])
    artifacts_produced["candidate_confusion_matrix.json"] = _sha256_file(output / "candidate_confusion_matrix.json")
    artifacts_produced["utility_weighted_confusion_matrix.json"] = _sha256_file(output / "utility_weighted_confusion_matrix.json")
    for name, cm in confusion["candidate_confusion_matrix"].get("policies", {}).items():
        acc = cm.get("accuracy", 0)
        print(f"  {name}: accuracy={acc:.3f}")

    # 9. Family and archetype analysis.
    print("\n[9/20] Analyzing families and archetypes...")
    from .family_analysis import analyze_families
    family_res = analyze_families(run, task_ids=test_ids)
    _write_json(output / "family_analysis.json", family_res["family_analysis"])
    _write_json(output / "archetype_analysis.json", family_res["archetype_analysis"])
    artifacts_produced["family_analysis.json"] = _sha256_file(output / "family_analysis.json")
    artifacts_produced["archetype_analysis.json"] = _sha256_file(output / "archetype_analysis.json")
    for fam, data in family_res["family_analysis"].get("families", {}).items():
        n = data.get("n_tasks", 0)
        cu = data.get("policy_utilities", {}).get("candidate", 0)
        print(f"  {fam}: n={n}, candidate={cu:.4f}")

    # 10. Feature audit.
    print("\n[10/20] Auditing features...")
    from .feature_audit import audit_features
    feat_audit = audit_features(run)
    _write_json(output / "feature_audit.json", feat_audit["feature_audit"])
    _write_json(output / "feature_ablation_results.json", feat_audit["feature_ablation_results"])
    artifacts_produced["feature_audit.json"] = _sha256_file(output / "feature_audit.json")
    artifacts_produced["feature_ablation_results.json"] = _sha256_file(output / "feature_ablation_results.json")
    rejected = feat_audit["feature_audit"].get("rejected_features", [])
    print(f"  Rejected features: {rejected}")

    # 11. Learning curves.
    print("\n[11/20] Computing learning curves...")
    from .learning_curves import compute_learning_curves
    curves = compute_learning_curves(run, n_seeds=20, seed=42)
    _write_json(output / "learning_curves.json", curves)
    artifacts_produced["learning_curves.json"] = _sha256_file(output / "learning_curves.json")
    print(f"  Trend decision: {curves.get('decision', curves.get('trend_decision', 'INCONCLUSIVE'))}")

    # 12. Learner comparison.
    print("\n[12/20] Comparing learners...")
    from .learner_comparison import compare_learners
    learners = compare_learners(run, seed=42)
    _write_json(output / "learner_comparison.json", learners)
    artifacts_produced["learner_comparison.json"] = _sha256_file(output / "learner_comparison.json")
    for name, data in learners.get("learners", {}).items():
        u = data.get("test_utility", 0)
        print(f"  {name}: test_utility={u:.4f}")

    # 13. Utility objective comparison.
    print("\n[13/20] Comparing utility objectives...")
    from .utility_objective import compare_objectives
    objectives = compare_objectives(run, seed=42)
    _write_json(output / "utility_objective_comparison.json", objectives)
    artifacts_produced["utility_objective_comparison.json"] = _sha256_file(output / "utility_objective_comparison.json")

    # 14. Action competence analysis.
    print("\n[14/20] Analyzing action competence...")
    from .competence import analyze_competence
    competence = analyze_competence(run, task_ids=test_ids)
    _write_json(output / "action_competence_analysis.json", competence)
    artifacts_produced["action_competence_analysis.json"] = _sha256_file(output / "action_competence_analysis.json")
    print(f"  ADI={competence.get('ADI', 0):.4f}, NADI={competence.get('NADI', 0):.4f}")

    # 15. Capability-routing decomposition.
    print("\n[15/20] Decomposing capability vs routing...")
    from .decomposition import decompose_capability_routing
    decomp = decompose_capability_routing(run, task_ids=test_ids, seed=42)
    _write_json(output / "capability_routing_decomposition.json", decomp)
    artifacts_produced["capability_routing_decomposition.json"] = _sha256_file(output / "capability_routing_decomposition.json")
    print(f"  C={decomp['model_capability_C']:.4f}, H={decomp['routing_headroom_H']:.4f}, R={decomp['router_recovery_R']:.4f}")

    # 16. Variance and influence analysis.
    print("\n[16/20] Analyzing variance and influence...")
    from .variance import analyze_variance
    variance = analyze_variance(run, task_ids=test_ids, seed=42)
    _write_json(output / "variance_analysis.json", variance["variance_analysis"])
    _write_json(output / "influence_analysis.json", variance["influence_analysis"])
    artifacts_produced["variance_analysis.json"] = _sha256_file(output / "variance_analysis.json")
    artifacts_produced["influence_analysis.json"] = _sha256_file(output / "influence_analysis.json")

    # 17. Cluster-aware CIs.
    print("\n[17/20] Computing cluster-aware confidence intervals...")
    from .cluster_bootstrap import compute_cluster_cis
    cluster_cis = compute_cluster_cis(run, task_ids=test_ids, seed=42)
    _write_json(output / "cluster_bootstrap_results.json", cluster_cis)
    artifacts_produced["cluster_bootstrap_results.json"] = _sha256_file(output / "cluster_bootstrap_results.json")
    print(f"  Cluster sensitive: {cluster_cis.get('cluster_sensitive', False)}")

    # 18. Power analysis.
    print("\n[18/20] Computing power analysis...")
    from .power_analysis import compute_power_analysis
    power = compute_power_analysis(run, task_ids=test_ids, seed=42)
    _write_json(output / "power_analysis.json", power)
    artifacts_produced["power_analysis.json"] = _sha256_file(output / "power_analysis.json")

    # 19. Seed stability.
    print("\n[19/20] Computing seed stability...")
    from .seed_stability import compute_seed_stability
    stability = compute_seed_stability(run, n_seeds=20, seed=42)
    _write_json(output / "seed_stability_results.json", stability)
    artifacts_produced["seed_stability_results.json"] = _sha256_file(output / "seed_stability_results.json")

    # 20. Remaining modules.
    print("\n[20/20] Running remaining analyses...")

    # Calibration audit.
    from .calibration import audit_calibration
    calib = audit_calibration(run, task_ids=test_ids)
    _write_json(output / "calibration_audit.json", calib)
    artifacts_produced["calibration_audit.json"] = _sha256_file(output / "calibration_audit.json")

    # Cross-run comparison.
    if mini_dir:
        from .cross_run import compare_runs
        cross = compare_runs(mini_dir, artifacts_dir)
        _write_json(output / "mini_full_comparison.json", cross)
        artifacts_produced["mini_full_comparison.json"] = _sha256_file(output / "mini_full_comparison.json")

    # Counterfactual equivalence.
    from .equivalence import verify_equivalence
    equiv = verify_equivalence(run)
    _write_json(output / "counterfactual_equivalence_report.json", equiv)
    artifacts_produced["counterfactual_equivalence_report.json"] = _sha256_file(output / "counterfactual_equivalence_report.json")

    # Benchmark identifiability.
    from .identifiability import audit_identifiability
    ident = audit_identifiability(run)
    _write_json(output / "benchmark_identifiability.json", ident)
    artifacts_produced["benchmark_identifiability.json"] = _sha256_file(output / "benchmark_identifiability.json")

    # Routing labels.
    from .routing_labels import classify_routing_labels
    labels = classify_routing_labels(run)
    _write_json(output / "routing_label_status.json", labels)
    artifacts_produced["routing_label_status.json"] = _sha256_file(output / "routing_label_status.json")

    # Gate A layered sub-gates.
    from .gate_a_layers import evaluate_gate_a_layers
    gate_a = evaluate_gate_a_layers(
        run, headroom_analysis=headroom, margin_analysis=margins,
        feature_audit=feat_audit.get("feature_audit"), task_ids=test_ids,
    )
    _write_json(output / "gate_a_layered_result.json", gate_a)
    artifacts_produced["gate_a_layered_result.json"] = _sha256_file(output / "gate_a_layered_result.json")
    print(f"  Gate A layered: {gate_a['gate_a_layered']['overall_status']}")
    for name, gate in gate_a["gate_a_layered"]["sub_gates"].items():
        print(f"    {name}: {gate['status']}")

    # Model scale decision.
    from .scale_decision import make_scale_decision
    scale_decision = make_scale_decision(
        run, headroom, sham_dist, feat_audit.get("feature_audit"), equiv, curves)
    _write_json(output / "model_scale_decision.json", scale_decision)
    artifacts_produced["model_scale_decision.json"] = _sha256_file(output / "model_scale_decision.json")
    print(f"  Scale decision: {scale_decision['decision']}")

    # Generate GATE_A_DIAGNOSTIC_REPORT.md
    from .report_generator import generate_diagnostic_report
    report = generate_diagnostic_report(
        run, headroom, margins, sham_audit, sham_dist, feat_audit,
        confusion, family_res, curves, learners, objectives,
        competence, decomp, variance, cluster_cis, power, stability,
        calib, gate_a, scale_decision, labels, ident, equiv,
    )
    report_path = output / "GATE_A_DIAGNOSTIC_REPORT.md"
    report_path.write_text(report)
    artifacts_produced["GATE_A_DIAGNOSTIC_REPORT.md"] = _sha256_file(report_path)

    # Write analysis manifest.
    elapsed = time.time() - start_time
    manifest = {
        "analysis_run_id": run_id,
        "source_run_id": run_id,
        "commit_sha": commit_sha,
        "package_version": "2.15.6",
        "qualification_version": "0.4.2",
        "original_counterfactual_artifact_hash": run.counterfactual_results_sha256,
        "analysis_code_source_hash": _sha256_json({"config": config_digest}),
        "configuration_digest": config_digest,
        "random_seeds": {"bootstrap": 42, "sham_base": 10000, "candidate_base": 20000},
        "artifacts_produced": artifacts_produced,
        "n_artifacts": len(artifacts_produced),
        "elapsed_seconds": elapsed,
        "timestamp": _timestamp(),
    }
    _write_json(output / "analysis_manifest.json", manifest)

    print(f"\n{'=' * 70}")
    print(f"v0.4.2 Analysis Complete — {len(artifacts_produced)} artifacts produced")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {output}")
    print(f"{'=' * 70}")

    return manifest


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _timestamp() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
