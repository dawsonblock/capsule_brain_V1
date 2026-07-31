"""Generate the GATE_A_DIAGNOSTIC_REPORT.md from all analysis results."""
from __future__ import annotations

from typing import Any

from .data import LoadedRun


def _f(v: Any) -> float:
    """Safely convert any value to float (0.0 for None/dict/list)."""
    if v is None or isinstance(v, (dict, list)):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def generate_diagnostic_report(
    run: LoadedRun,
    headroom: dict[str, Any],
    margins: dict[str, Any],
    sham_audit: dict[str, Any],
    sham_dist: dict[str, Any],
    feature_audit: dict[str, Any],
    confusion: dict[str, Any],
    family_res: dict[str, Any],
    learning_curves: dict[str, Any],
    learners: dict[str, Any],
    objectives: dict[str, Any],
    competence: dict[str, Any],
    decomposition: dict[str, Any],
    variance: dict[str, Any],
    cluster_cis: dict[str, Any],
    power: dict[str, Any],
    stability: dict[str, Any],
    calibration: dict[str, Any],
    gate_a: dict[str, Any],
    scale_decision: dict[str, Any],
    routing_labels: dict[str, Any],
    identifiability: dict[str, Any],
    equivalence: dict[str, Any],
) -> str:
    """Generate the full diagnostic report as markdown."""
    lines = []
    w = lines.append

    w("# Gate A Diagnostic Report — AutoLearn v0.4.2")
    w("")
    w(f"**Package**: 2.15.6 | **Qualification**: 0.4.2 | **Model**: {run.qualification_report.get('provider', 'unknown')}")
    w(f"**Tasks**: {len(run.tasks)} | **Counterfactuals**: {len(run.outcomes)} | **Test tasks**: {len(run.test_task_ids)}")
    w("")
    w("---")
    w("")

    # Section 1: Conclusion
    w("## 1. Conclusion")
    w("")
    overall = gate_a.get("gate_a_layered", {}).get("overall_status", "UNKNOWN")
    w(f"**Gate A Layered Status**: {overall}")
    w("")
    w("### Preregistered Result (from 7B full run)")
    w("")
    w("> On the 130-task, 490-counterfactual 7B run, the candidate improved mean")
    w("> utility over the frozen baseline by 0.415, but the paired-bootstrap lower")
    w("> confidence bound of 0.004 did not exceed the predeclared practical")
    w("> threshold of 0.01. The candidate also failed to outperform the sham")
    w("> policy, with a mean candidate-minus-sham delta of -0.020. The experiment")
    w("> therefore did not demonstrate state-dependent causal routing improvement")
    w("> beyond marginal action frequency.")
    w("")

    # Section 2: Gate A Layered Sub-Gates
    w("## 2. Gate A Layered Sub-Gates")
    w("")
    w("| Sub-Gate | Status | Key Metric |")
    w("|----------|--------|------------|")
    for name, gate in gate_a.get("gate_a_layered", {}).get("sub_gates", {}).items():
        status = gate.get("status", "UNKNOWN")
        metric = ""
        if "ci_low" in gate:
            metric = f"LCB={gate['ci_low']:.4f}"
        elif "H_oracle_sham" in gate:
            metric = f"H={gate['H_oracle_sham']:.4f}"
        elif "recovery" in gate:
            metric = f"R={gate['recovery']:.4f}"
        elif "beats_majority" in gate:
            metric = f"beats_majority={gate['beats_majority']}"
        w(f"| {name} | {status} | {metric} |")
    w("")

    # Section 3: Routing Headroom
    w("## 3. Oracle Routing Headroom")
    w("")
    w(f"| Quantity | Value |")
    w(f"|----------|-------|")
    w(f"| U_oracle | {headroom.get('U_oracle', 0):.4f} |")
    w(f"| U_sham | {headroom.get('U_sham', 0):.4f} |")
    w(f"| U_baseline | {headroom.get('U_baseline', 0):.4f} |")
    w(f"| U_candidate | {headroom.get('U_candidate', 0):.4f} |")
    w(f"| U_frequency | {headroom.get('U_frequency', 0):.4f} |")
    w(f"| H_oracle_sham | {headroom.get('H_oracle_sham', 0):.4f} |")
    w(f"| H_oracle_baseline | {headroom.get('H_oracle_baseline', 0):.4f} |")
    w(f"| R_sham | {headroom.get('R_sham', 0):.4f} |")
    w(f"| R_baseline | {headroom.get('R_baseline', 0):.4f} |")
    w(f"| **Status** | **{headroom.get('routing_headroom_status', 'UNKNOWN')}** |")
    w("")
    w(f"**Interpretation**: {headroom.get('interpretation', '')}")
    w("")

    # Section 4: Best-Action Imbalance
    w("## 4. Best-Action Imbalance")
    w("")
    w("See `best_action_distribution.json` for full details.")
    w("")

    # Section 5: Utility Margins
    w("## 5. Utility-Margin Analysis")
    w("")
    w("| Bucket | N | Candidate | Baseline | Sham | Oracle | C-Sham | C-Baseline |")
    w("|--------|---|-----------|----------|------|--------|--------|------------|")
    for bucket_name in ["near_tie", "weak", "meaningful", "high_value"]:
        bucket = margins.get("buckets", {}).get(bucket_name, {})
        _n = bucket.get("n_tasks", bucket.get("task_count", 0))
        _cu = _f(bucket.get("candidate_utility"))
        _bu = _f(bucket.get("baseline_utility"))
        _su = _f(bucket.get("sham_utility"))
        _ou = _f(bucket.get("oracle_utility"))
        _cs = _f(bucket.get("candidate_minus_sham"))
        _cb = _f(bucket.get("candidate_minus_baseline"))
        w(f"| {bucket_name} | {_n} | "
          f"{_cu:.3f} | "
          f"{_bu:.3f} | "
          f"{_su:.3f} | "
          f"{_ou:.3f} | "
          f"{_cs:.3f} | "
          f"{_cb:.3f} |")
    w("")
    hm = margins.get("Gate_A_high_margin", {})
    w(f"**Gate A (high-margin only)**: candidate={_f(hm.get('candidate_utility')):.3f}, "
      f"sham={_f(hm.get('sham_utility')):.3f}, oracle={_f(hm.get('oracle_utility')):.3f}")
    w("")

    # Section 6: Sham Audit
    w("## 6. Sham Construction Audit")
    w("")
    w(f"- Sham seeds: {sham_dist.get('n_sham_seeds', 0)}")
    w(f"- Sham utility: mean={sham_dist.get('sham_mean', 0):.4f}, "
      f"std={sham_dist.get('sham_std', 0):.4f}")
    w(f"- Sham 95% CI: [{sham_dist.get('sham_ci_low', 0):.4f}, {sham_dist.get('sham_ci_high', 0):.4f}]")
    w(f"- Candidate utility: {sham_dist.get('candidate_utility', 0):.4f}")
    w(f"- Candidate percentile: {_f(sham_dist.get('candidate_percentile')):.1f}%")
    w(f"- Criterion met (candidate > sham upper 97.5%): {sham_dist.get('criterion_met', False)}")
    w("")

    # Section 7: Feature Audit
    w("## 7. Feature Audit")
    w("")
    fa = feature_audit.get("feature_audit", {})
    w(f"- Rejected features: {fa.get('rejected_features', [])}")
    w(f"- Feature count: {len(fa.get('features', {}))}")
    w("See `feature_audit.json` and `feature_ablation_results.json` for details.")
    w("")

    # Section 8: Learning Curves
    w("## 8. Learning Curves")
    w("")
    w(f"- Trend decision: {learning_curves.get('trend_decision', 'INCONCLUSIVE')}")
    w("See `learning_curves.json` for full curve data.")
    w("")

    # Section 9: Learner Comparison
    w("## 9. Learner Capacity Comparison")
    w("")
    w("| Learner | Test Utility |")
    w("|---------|-------------|")
    for name, data in learners.get("learners", {}).items():
        w(f"| {name} | {data.get('test_utility', 0):.4f} |")
    w("")

    # Section 10: Competence
    w("## 10. Action-Specific Competence")
    w("")
    w(f"- ADI: {competence.get('ADI', 0):.4f}")
    w(f"- NADI: {competence.get('NADI', 0):.4f}")
    w("")

    # Section 11: Capability-Routing Decomposition
    w("## 11. Capability vs Routing Decomposition")
    w("")
    w(f"| Quantity | Value |")
    w(f"|----------|-------|")
    w(f"| Model capability (C) | {decomposition.get('model_capability_C', 0):.4f} |")
    w(f"| Routing headroom (H) | {decomposition.get('routing_headroom_H', 0):.4f} |")
    w(f"| Router recovery (R) | {decomposition.get('router_recovery_R', 0):.4f} |")
    w("")
    w(f"**Interpretation**: {decomposition.get('interpretation', '')}")
    w("")

    # Section 12: Variance
    w("## 12. Variance and Influence Analysis")
    w("")
    va = variance.get("variance_analysis", {})
    cb = va.get("candidate_baseline", {})
    w(f"| Statistic | Candidate-Baseline | Candidate-Sham |")
    w(f"|-----------|-------------------|----------------|")
    cs = va.get("candidate_sham", {})
    for stat in ["mean", "median", "std", "min", "max", "p5", "p95"]:
        w(f"| {stat} | {cb.get(stat, 0):.4f} | {cs.get(stat, 0):.4f} |")
    w("")

    # Section 13: Cluster Bootstrap
    w("## 13. Cluster-Aware Confidence Intervals")
    w("")
    w(f"- Cluster sensitive: {cluster_cis.get('cluster_sensitive', False)}")
    w(f"- Primary method: {cluster_cis.get('primary_method', 'archetype_cluster')}")
    w("See `cluster_bootstrap_results.json` for all interval variants.")
    w("")

    # Section 14: Power Analysis
    w("## 14. Power and Sample-Size Analysis")
    w("")
    w("See `power_analysis.json` for required task counts at each effect size.")
    w("")

    # Section 15: Seed Stability
    w("## 15. Seed Stability")
    w("")
    w("See `seed_stability_results.json` for 20-seed distributions.")
    w("")

    # Section 16: Scale Decision
    w("## 16. Model-Scale Decision")
    w("")
    w(f"**Decision**: {scale_decision.get('decision', 'UNKNOWN')}")
    w(f"**Reason**: {scale_decision.get('reason', '')}")
    w("")
    w("### Blocking Conditions")
    w("")
    for cond, val in scale_decision.get("blocking_conditions_checked", {}).items():
        w(f"- {cond}: {val}")
    w("")

    # Section 17: Routing Labels
    w("## 17. Routing Label Status")
    w("")
    w("See `routing_label_status.json` for label proportions.")
    w("")

    # Section 18: Identifiability
    w("## 18. Benchmark Identifiability")
    w("")
    w("See `benchmark_identifiability.json` for family-only vs full-feature comparison.")
    w("")

    # Section 19: Counterfactual Equivalence
    w("## 19. Counterfactual Equivalence")
    w("")
    w(f"- Equivalence verified: {equivalence.get('equivalence_verified', False)}")
    w(f"- Excluded tasks: {equivalence.get('excluded_count', 0)}")
    w("")

    # Section 20: Decision Tree
    w("## 20. Decision Tree Classification")
    w("")
    H = headroom.get("H_oracle_sham", 0)
    R_sham = headroom.get("R_sham", 0)
    if H < 0.10:
        w("**Case 1**: Oracle approximately equals sham.")
        w("- Conclusion: Routing headroom is insufficient.")
        w("- Action: Redesign benchmark and action conditions. Do not scale model.")
    elif H > 0.50 and R_sham <= 0:
        w("**Case 2**: Oracle strongly beats sham, but candidate approximately equals sham.")
        w("- Conclusion: Available routing signal is not being learned.")
        w("- Action: Improve features, labels, objective or learner. Do not scale model yet.")
    elif R_sham > 0:
        w("**Case 3+**: Candidate shows some recovery.")
        w("- Action: Continue analysis of high-margin tasks and data sufficiency.")
    else:
        w("**Inconclusive**: Need more analysis to classify.")
    w("")

    # Section 21: Allowed Conclusion Language
    w("## 21. Allowed Conclusion Language")
    w("")
    w("This report distinguishes preregistered results from retrospective diagnostics.")
    w("The preregistered primary metric is the full Gate A result from the 7B run.")
    w("All v0.4.2 analyses are retrospective diagnostics that do not replace the")
    w("preregistered result.")
    w("")
    w("No unsupported claim attributes failure primarily to model size.")
    w("A 14B run is not recommended without routing-headroom evidence.")
    w("")
    w("---")
    w("*Generated by AutoLearn v0.4.2 diagnostic pipeline.*")

    return "\n".join(lines)
