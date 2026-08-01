"""v0.4.7 final qualification report generator.

Generates ``FINAL_REPORT.md`` — the authoritative v0.4.7 report.

Language rules (enforced here):
    - Do NOT say "Gate A passed" unless Gate A2 AND Gate A3 pass.
    - Use precise claims: "Gate A0 evidence admissibility passed."
    - Synthetic fixtures must be labeled in title and first paragraph.
    - Historical external evidence not shipped must be labeled:
      "UNVERIFIED HISTORICAL CLAIM — SOURCE ARTIFACTS NOT INCLUDED"
    - Use full ISO dates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from capsule_brain.version import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
)


# Allowed report language — must not be paraphrased.
SYNTHETIC_LABEL = (
    "SYNTHETIC EVIDENCE — PIPELINE VALIDATION ONLY — NOT A REAL-MODEL RESULT"
)

UNAVAILABLE_HISTORICAL_LABEL = (
    "UNVERIFIED HISTORICAL CLAIM — SOURCE ARTIFACTS NOT INCLUDED"
)

REAL_MODEL_LABEL = "REAL-MODEL EVIDENCE"


def _sym(status: str) -> str:
    """Return the status symbol unchanged (no mapping)."""
    return status if status else "BLOCKED"


def _status(d: dict | None, key: str = "status", default: str = "BLOCKED") -> str:
    if d is None or not isinstance(d, dict):
        return default
    return d.get(key, default)


def _reasons(d: dict | None) -> list[str]:
    if d is None or not isinstance(d, dict):
        return []
    r = d.get("reasons", [])
    if isinstance(r, list):
        return r
    return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_float(v: Any, digits: int = 6) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _comparison_table(comp: dict | None, title: str, lines: list[str]) -> None:
    """Render a comparison-result dict as a markdown table."""
    w = lines.append
    if not comp or not isinstance(comp, dict):
        w(f"**{title}**: BLOCKED — no comparison data available.")
        w("")
        return
    w(f"### {title}")
    w("")
    w(f"**Status**: {'PASS' if comp.get('passes') else 'FAIL'}")
    w("")
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| n_tasks | {comp.get('n_tasks', 0)} |")
    w(f"| n_task_groups | {comp.get('n_task_groups', 0)} |")
    w(f"| mean_delta | {_fmt_float(comp.get('mean_delta'))} |")
    w(f"| median_delta | {_fmt_float(comp.get('median_delta'))} |")
    w(f"| standard_error | {_fmt_float(comp.get('standard_error'))} |")
    w(f"| confidence_level | {_fmt_float(comp.get('confidence_level'), 2)} |")
    w(f"| lower_bound | {_fmt_float(comp.get('lower_bound'))} |")
    w(f"| upper_bound | {_fmt_float(comp.get('upper_bound'))} |")
    w(f"| practical_threshold | {_fmt_float(comp.get('practical_threshold'))} |")
    w(f"| passes | {comp.get('passes', False)} |")
    w(f"| win_rate | {_fmt_float(comp.get('win_rate'), 4)} |")
    w(f"| tie_rate | {_fmt_float(comp.get('tie_rate'), 4)} |")
    w(f"| loss_rate | {_fmt_float(comp.get('loss_rate'), 4)} |")
    w("")


def generate_v047_report(
    verdict: dict,  # QualificationVerdict.to_dict()
    config: dict,
    evidence_summary: dict,
    family_evaluation: dict,
    collapse_check: dict,
    safety_check: dict,
    checksums: dict,
) -> str:
    """Generate the v0.4.7 final qualification report as markdown.

    Required sections (per spec):
    1. Run identity
    2. Source identity
    3. Model and provider identity
    4. Benchmark and split summary
    5. Counterfactual execution summary
    6. Evidence-admissibility results (Gate A0)
    7. Routing-headroom results (Gate A1)
    8. Candidate-vs-baseline results (Gate A2)
    9. Candidate-vs-sham results (Gate A2)
    10. Replication results (Gate A3)
    11. Family-level results
    12. Safety results
    13. Cost and latency results
    14. Action-distribution results
    15. Artifact-integrity results
    16. Promotion decision (Gate A4)
    17. Limitations
    18. Exact commands to reproduce
    19. Checksums

    Language rules:
    - Do NOT say "Gate A passed" unless Gate A2 AND Gate A3 pass.
    - Use precise claims: "Gate A0 evidence admissibility passed."
    - Synthetic fixtures must be labeled in title and first paragraph.
    - Historical external evidence not shipped must be labeled:
      "UNVERIFIED HISTORICAL CLAIM — SOURCE ARTIFACTS NOT INCLUDED"
    - Use full ISO dates.
    """
    lines: list[str] = []
    w = lines.append

    evidence_origin = str(verdict.get("evidence_origin", "")).upper()
    is_synthetic = evidence_origin == "SYNTHETIC"
    is_unavailable = evidence_origin == "UNAVAILABLE"

    # Determine the prominent label.
    if is_synthetic:
        origin_label = SYNTHETIC_LABEL
    elif is_unavailable:
        origin_label = UNAVAILABLE_HISTORICAL_LABEL
    else:
        origin_label = REAL_MODEL_LABEL

    # --- Title ---
    if is_synthetic:
        w("# v0.4.7 Final Qualification Report — SYNTHETIC EVIDENCE")
    elif is_unavailable:
        w("# v0.4.7 Final Qualification Report — UNVERIFIED HISTORICAL CLAIM")
    else:
        w("# v0.4.7 Final Qualification Report")
    w("")
    w(f"**{origin_label}**")
    w("")

    # --- First paragraph with label ---
    if is_synthetic:
        w(
            "The synthetic routing fixture is structurally complete and "
            "suitable for testing evidence schemas, policy evaluation, Gate A "
            "arithmetic and audit plumbing. It is not real-model evidence, is "
            "not scientifically claim-eligible, cannot be promoted and cannot "
            "support model-scale decisions."
        )
    elif is_unavailable:
        w(
            "The original source artifacts are not available in this "
            "repository. The historical Gate A result cannot be reproduced "
            "from the current source package. Synthetic fixtures are "
            "maintained separately and are not used as substitutes."
        )
    else:
        w(
            "This report summarizes the v0.4.7 qualification run with "
            "real-model evidence. All gate results below are computed from "
            "the evidence in the run directory."
        )
    w("")

    run_id = verdict.get("run_id", "unknown")
    protocol_version = verdict.get("protocol_version", "0.4.7")

    # Gate statuses for convenience.
    gate_a0 = verdict.get("gate_a0_admissibility", {})
    gate_a1 = verdict.get("gate_a1_headroom", {})
    gate_a2 = verdict.get("gate_a2_effectiveness", {})
    gate_a3 = verdict.get("gate_a3_robustness", {})
    gate_a4 = verdict.get("gate_a4_promotion", {})

    a0_status = _status(gate_a0)
    a1_status = _status(gate_a1)
    a2_status = _status(gate_a2)
    a3_status = _status(gate_a3)
    a4_status = _status(gate_a4)

    # Legacy gate A status (conservative: PASS only when A2 AND A3 pass).
    a2_pass = a2_status == "PASS"
    a3_pass = a3_status == "PASS"
    legacy_gate_a = "PASS" if (a2_pass and a3_pass) else (
        "BLOCKED" if (a2_status == "BLOCKED" or a3_status == "BLOCKED")
        else "FAIL"
    )

    # --- Section 1: Run identity ---
    w("## 1. Run Identity")
    w("")
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Run ID | {run_id} |")
    w(f"| Protocol version | {protocol_version} |")
    w(f"| Package version | {PACKAGE_VERSION} |")
    w(f"| AutoLearn version | {AUTOLEARN_VERSION} |")
    w(f"| Qualification version | {AUTOLEARN_QUALIFICATION_VERSION} |")
    w(f"| Evidence origin | {evidence_origin or 'unknown'} |")
    w(f"| Report generated | {_now_iso()} |")
    w("")

    # --- Section 2: Source identity ---
    w("## 2. Source Identity")
    w("")
    source = evidence_summary.get("source_manifest", {})
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Source identity method | {source.get('source_identity_method', 'unknown')} |")
    git_commit = source.get("git_commit")
    w(f"| Git commit | {git_commit if git_commit else 'N/A'} |")
    w(f"| Git dirty | {source.get('git_dirty', 'N/A')} |")
    tree_digest = source.get("tree_digest", "")
    if tree_digest:
        w(f"| Tree digest (SHA-256) | {tree_digest[:16]}... |")
    else:
        w("| Tree digest (SHA-256) | N/A |")
    w(f"| File count | {source.get('file_count', 0)} |")
    w("")

    # --- Section 3: Model and provider identity ---
    w("## 3. Model and Provider Identity")
    w("")
    model_manifest = evidence_summary.get("model_manifest", {})
    provider_manifest = evidence_summary.get("provider_manifest", {})
    tokenizer_manifest = evidence_summary.get("tokenizer_manifest", {})
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Model ID | {model_manifest.get('model_id', config.get('model_id', ''))} |")
    w(f"| Model revision | {model_manifest.get('revision', config.get('model_revision', ''))} |")
    w(f"| Tokenizer ID | {tokenizer_manifest.get('tokenizer_id', model_manifest.get('model_id', ''))} |")
    w(f"| Tokenizer revision | {tokenizer_manifest.get('revision', config.get('tokenizer_revision', ''))} |")
    w(f"| Dtype | {config.get('dtype', '')} |")
    w(f"| Device | {config.get('device', '')} |")
    w(f"| Provider class | {provider_manifest.get('provider_class', '')} |")
    w(f"| Runtime type | {provider_manifest.get('runtime_type', '')} |")
    w("")

    # --- Section 4: Benchmark and split summary ---
    w("## 4. Benchmark and Split Summary")
    w("")
    benchmark = evidence_summary.get("benchmark_manifest", {})
    split = evidence_summary.get("split_manifest", {})
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Benchmark name | {benchmark.get('benchmark_name', '')} |")
    w(f"| Benchmark task count | {benchmark.get('task_count', 0)} |")
    w(f"| Split seed | {config.get('split_seed', '')} |")
    w(f"| Experience fraction | {_fmt_float(config.get('experience_fraction'), 2)} |")
    w(f"| Validation fraction | {_fmt_float(config.get('validation_fraction'), 2)} |")
    w(f"| Test fraction | {_fmt_float(config.get('test_fraction'), 2)} |")
    w(f"| OOD fraction | {_fmt_float(config.get('ood_fraction'), 2)} |")
    w(f"| Group by | {config.get('group_by', '')} |")
    w(f"| Test task count | {split.get('test_task_count', 0)} |")
    w(f"| Test task group count | {split.get('test_task_group_count', 0)} |")
    w("")

    # --- Section 5: Counterfactual execution summary ---
    w("## 5. Counterfactual Execution Summary")
    w("")
    cf = evidence_summary.get("counterfactual_summary", {})
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Tasks with counterfactual outcomes | {cf.get('n_tasks', 0)} |")
    w(f"| Total counterfactual outcomes | {cf.get('n_outcomes', 0)} |")
    w(f"| Counterfactual equivalence verified | {cf.get('equivalence_verified', False)} |")
    w(f"| Temperature | {_fmt_float(config.get('temperature'), 1)} |")
    w(f"| do_sample | {config.get('do_sample', False)} |")
    w(f"| Max new tokens | {config.get('max_new_tokens', 0)} |")
    w(f"| Timeout seconds | {config.get('timeout_seconds', 0)} |")
    w("")

    # --- Section 6: Evidence-admissibility results (Gate A0) ---
    w("## 6. Evidence-Admissibility Results (Gate A0)")
    w("")
    w(f"**Gate A0 status**: {_sym(a0_status)}")
    w("")
    if a0_status == "PASS":
        w("Gate A0 evidence admissibility passed.")
    elif a0_status == "BLOCKED":
        w("Gate A0 evidence admissibility is BLOCKED — required evidence is missing.")
    else:
        w(f"Gate A0 evidence admissibility {_sym(a0_status).lower()}.")
    w("")
    checks = gate_a0.get("checks", {}) if isinstance(gate_a0, dict) else {}
    if checks:
        w(f"**Sub-gates**: {checks.get('n_sub_gates', 0)}")
        w(f"**Pass**: {checks.get('n_pass', 0)}")
        w(f"**Fail**: {checks.get('n_fail', 0)}")
        w(f"**Blocked**: {checks.get('n_blocked', 0)}")
        w(f"**Not applicable**: {checks.get('n_not_applicable', 0)}")
        sub_gates = checks.get("sub_gates", {})
        if isinstance(sub_gates, dict) and sub_gates:
            w("")
            w("| Sub-gate | Status | Reason |")
            w("|----------|--------|--------|")
            for sg_name in sorted(sub_gates.keys()):
                sg = sub_gates[sg_name]
                if isinstance(sg, dict):
                    w(f"| {sg_name} | {sg.get('status', '')} | {sg.get('reason', '')} |")
    a0_reasons = _reasons(gate_a0)
    if a0_reasons:
        w("")
        w("**Reasons**:")
        for r in a0_reasons:
            w(f"- {r}")
    w("")
    w(
        "*Note: Gate A0 PASS does NOT imply Gate A effectiveness. "
        "Gate A0 only determines that evidence is admissible for evaluation.*"
    )
    w("")

    # --- Section 7: Routing-headroom results (Gate A1) ---
    w("## 7. Routing-Headroom Results (Gate A1)")
    w("")
    w(f"**Gate A1 status**: {_sym(a1_status)}")
    w("")
    if a1_status == "PASS":
        w("Gate A1 routing headroom passed.")
    elif a1_status == "BLOCKED":
        w("Gate A1 routing headroom is BLOCKED — no paired task rows available.")
    else:
        w(f"Gate A1 routing headroom {_sym(a1_status).lower()}.")
    w("")
    if isinstance(gate_a1, dict):
        _comparison_table(gate_a1.get("oracle_vs_baseline"), "Oracle vs Baseline", lines)
        _comparison_table(gate_a1.get("oracle_vs_sham"), "Oracle vs Sham", lines)
    a1_reasons = _reasons(gate_a1)
    if a1_reasons:
        w("**Reasons**:")
        for r in a1_reasons:
            w(f"- {r}")
        w("")
    w(
        "*Note: A Gate A1 failure means the benchmark offers insufficient "
        "routing headroom. It does NOT mean the learner is defective.*"
    )
    w("")

    # --- Section 8: Candidate-vs-baseline results (Gate A2) ---
    w("## 8. Candidate-vs-Baseline Results (Gate A2)")
    w("")
    if isinstance(gate_a2, dict):
        _comparison_table(gate_a2.get("candidate_vs_baseline"), "Candidate vs Baseline", lines)
    else:
        w("**Candidate vs Baseline**: BLOCKED — no comparison data available.")
        w("")
    w(
        "*Pass criterion: LCB(delta_candidate_vs_baseline) > epsilon_0. "
        "A zero or positive point estimate must NOT pass when the lower "
        "confidence bound does not exceed the threshold.*"
    )
    w("")

    # --- Section 9: Candidate-vs-sham results (Gate A2) ---
    w("## 9. Candidate-vs-Sham Results (Gate A2)")
    w("")
    w(f"**Gate A2 status**: {_sym(a2_status)}")
    w("")
    if a2_status == "PASS":
        w("Gate A2 candidate causal effectiveness passed.")
    elif a2_status == "BLOCKED":
        w("Gate A2 candidate causal effectiveness is BLOCKED — no paired task rows available.")
    else:
        w(f"Gate A2 candidate causal effectiveness {_sym(a2_status).lower()}.")
    w("")
    if isinstance(gate_a2, dict):
        _comparison_table(gate_a2.get("candidate_vs_sham"), "Candidate vs Sham", lines)
    else:
        w("**Candidate vs Sham**: BLOCKED — no comparison data available.")
        w("")
    a2_reasons = _reasons(gate_a2)
    if a2_reasons:
        w("**Reasons**:")
        for r in a2_reasons:
            w(f"- {r}")
        w("")
    w(
        "*The candidate must beat BOTH baseline and sham. Gate A2 passes "
        "only when both comparisons' LCBs exceed their respective thresholds.*"
    )
    w("")

    # --- Section 10: Replication results (Gate A3) ---
    w("## 10. Replication Results (Gate A3)")
    w("")
    w(f"**Gate A3 status**: {_sym(a3_status)}")
    w("")
    if a3_status == "PASS":
        w("Gate A3 robustness and replication passed.")
    elif a3_status == "BLOCKED":
        w("Gate A3 robustness and replication is BLOCKED.")
    else:
        w(f"Gate A3 robustness and replication {_sym(a3_status).lower()}.")
    w("")
    if isinstance(gate_a3, dict):
        rep = gate_a3.get("replicate_summary", {})
        w("| Field | Value |")
        w("|-------|-------|")
        w(f"| n_replicates | {rep.get('n_replicates', 0)} |")
        w(f"| n_generation_seeds | {rep.get('n_generation_seeds', 0)} |")
        w(f"| n_learner_seeds | {rep.get('n_learner_seeds', 0)} |")
        w(f"| Replication pass rate | {_fmt_float(rep.get('replication_pass_rate'), 4)} |")
        w(f"| n_replicates_passing | {rep.get('n_replicates_passing', 0)} |")
        cat_reversals = rep.get("catastrophic_reversals", [])
        w(f"| Catastrophic reversals | {len(cat_reversals)} |")
        w("")
        if cat_reversals:
            w("**Catastrophic reversals**:")
            for cr in cat_reversals:
                w(f"- {cr}")
            w("")
    a3_reasons = _reasons(gate_a3)
    if a3_reasons:
        w("**Reasons**:")
        for r in a3_reasons:
            w(f"- {r}")
        w("")
    w(
        "*Gate A3 validates seed diversity, sign-reversal bounds, "
        "replication pass rate, family-stratified evaluation, "
        "action-distribution collapse, and safety.*"
    )
    w("")

    # --- Section 11: Family-level results ---
    w("## 11. Family-Level Results")
    w("")
    if family_evaluation and isinstance(family_evaluation, dict):
        w(f"**Sufficient families**: {family_evaluation.get('n_sufficient', 0)}")
        w(f"**Insufficient families**: {family_evaluation.get('n_insufficient', 0)}")
        w(f"**Min test tasks**: {family_evaluation.get('min_test_tasks', 0)}")
        w(f"**Min test groups**: {family_evaluation.get('min_test_groups', 0)}")
        w("")
        families = family_evaluation.get("families", {})
        if isinstance(families, dict) and families:
            w("| Family | Status | Tasks | Groups | Cand-Baseline Delta | Cand-Sham Delta | CB LCB | CS LCB |")
            w("|--------|--------|-------|--------|---------------------|-----------------|--------|--------|")
            for fam_name in sorted(families.keys()):
                f = families[fam_name]
                if isinstance(f, dict):
                    w(
                        f"| {fam_name} | {f.get('status', '')} "
                        f"| {f.get('task_count', 0)} | {f.get('task_group_count', 0)} "
                        f"| {_fmt_float(f.get('candidate_baseline_delta'))} "
                        f"| {_fmt_float(f.get('candidate_sham_delta'))} "
                        f"| {_fmt_float(f.get('candidate_baseline_lcb'))} "
                        f"| {_fmt_float(f.get('candidate_sham_lcb'))} |"
                    )
        critical = family_evaluation.get("critical_regressions", [])
        if critical:
            w("")
            w("**Critical regressions**:")
            for cr in critical:
                w(f"- {cr}")
    else:
        w("**Family evaluation**: BLOCKED — no family evaluation data available.")
    w("")

    # --- Section 12: Safety results ---
    w("## 12. Safety Results")
    w("")
    if safety_check and isinstance(safety_check, dict):
        w(f"**Safety status**: {_sym(_status(safety_check))}")
        w("")
        w("| Field | Value |")
        w("|-------|-------|")
        w(f"| n_safety_tasks | {safety_check.get('n_safety_tasks', 0)} |")
        w(f"| n_severe_violations_candidate | {safety_check.get('n_severe_violations_candidate', 0)} |")
        w(f"| n_severe_violations_baseline | {safety_check.get('n_severe_violations_baseline', 0)} |")
        w(f"| n_total_violations_candidate | {safety_check.get('n_total_violations_candidate', 0)} |")
        w(f"| n_total_violations_baseline | {safety_check.get('n_total_violations_baseline', 0)} |")
        w(f"| severe_violation_increase | {safety_check.get('severe_violation_increase', 0)} |")
        w(f"| total_violation_rate_increase | {_fmt_float(safety_check.get('total_violation_rate_increase'), 6)} |")
        w(f"| all_critical_cases_pass | {safety_check.get('all_critical_cases_pass', False)} |")
        s_reasons = safety_check.get("reasons", [])
        if s_reasons:
            w("")
            w("**Reasons**:")
            for r in s_reasons:
                w(f"- {r}")
    else:
        w("**Safety evaluation**: BLOCKED — no safety evaluation data available.")
    w("")
    w(
        "*Promotion must block if the candidate increases severe safety "
        "violations. Severe safety failures must NOT be averaged away "
        "through general utility improvements.*"
    )
    w("")

    # --- Section 13: Cost and latency results ---
    w("## 13. Cost and Latency Results")
    w("")
    cost = evidence_summary.get("cost_latency", {})
    if cost and isinstance(cost, dict):
        w("| Field | Value |")
        w("|-------|-------|")
        w(f"| Total generation time (s) | {_fmt_float(cost.get('total_generation_time_s', 0), 2)} |")
        w(f"| Mean generation time per task (s) | {_fmt_float(cost.get('mean_generation_time_per_task_s', 0), 2)} |")
        w(f"| Total tokens generated | {cost.get('total_tokens_generated', 0)} |")
        w(f"| Mean tokens per task | {_fmt_float(cost.get('mean_tokens_per_task', 0), 1)} |")
        w(f"| Timeout count | {cost.get('timeout_count', 0)} |")
        w(f"| Error count | {cost.get('error_count', 0)} |")
    else:
        w("**Cost and latency**: No cost/latency data available.")
    w("")

    # --- Section 14: Action-distribution results ---
    w("## 14. Action-Distribution Results")
    w("")
    if collapse_check and isinstance(collapse_check, dict):
        w(f"**Collapse check status**: {_sym(_status(collapse_check))}")
        w("")
        w("| Field | Value |")
        w("|-------|-------|")
        w(f"| Entropy (nats) | {_fmt_float(collapse_check.get('entropy'), 6)} |")
        w(f"| Max action share | {_fmt_float(collapse_check.get('max_action_share'), 4)} |")
        w(f"| Action coverage | {collapse_check.get('action_coverage', 0)} |")
        w(f"| JS divergence from baseline | {_fmt_float(collapse_check.get('js_divergence_from_baseline'), 6)} |")
        w(f"| Abstention share | {_fmt_float(collapse_check.get('abstention_share'), 4)} |")
        w(f"| Baseline abstention share | {_fmt_float(collapse_check.get('baseline_abstention_share'), 4)} |")
        w(f"| Abstention increase | {_fmt_float(collapse_check.get('abstention_increase'), 4)} |")
        w(f"| Invalid action rate | {_fmt_float(collapse_check.get('invalid_action_rate'), 4)} |")
        c_reasons = collapse_check.get("reasons", [])
        if c_reasons:
            w("")
            w("**Reasons**:")
            for r in c_reasons:
                w(f"- {r}")
    else:
        w("**Action-distribution**: BLOCKED — no collapse check data available.")
    w("")

    # --- Section 15: Artifact-integrity results ---
    w("## 15. Artifact-Integrity Results")
    w("")
    artifact_dag = evidence_summary.get("artifact_dag", {})
    if artifact_dag and isinstance(artifact_dag, dict):
        w(f"**DAG valid**: {artifact_dag.get('valid', False)}")
        issues = artifact_dag.get("issues", [])
        w(f"**Issues**: {len(issues)}")
        if issues:
            w("")
            for issue in issues:
                w(f"- {issue}")
    else:
        w("**Artifact DAG**: No artifact DAG validation data available.")
    w("")
    # Checksum verification.
    if checksums and isinstance(checksums, dict):
        w(f"**Checksums valid**: {checksums.get('valid', False)}")
        w(f"**Mismatches**: {len(checksums.get('mismatches', []))}")
        w(f"**Missing files**: {len(checksums.get('missing', []))}")
        w(f"**Extra files**: {len(checksums.get('extra', []))}")
        mismatches = checksums.get("mismatches", [])
        missing = checksums.get("missing", [])
        extra = checksums.get("extra", [])
        if mismatches:
            w("")
            w("**Checksum mismatches**:")
            for m in mismatches:
                w(f"- {m}")
        if missing:
            w("")
            w("**Missing files**:")
            for m in missing:
                w(f"- {m}")
        if extra:
            w("")
            w("**Extra files (not in checksum manifest)**:")
            for e in extra:
                w(f"- {e}")
    else:
        w("**Checksums**: No checksum verification data available.")
    w("")

    # --- Section 16: Promotion decision (Gate A4) ---
    w("## 16. Promotion Decision (Gate A4)")
    w("")
    w(f"**Gate A4 status**: {_sym(a4_status)}")
    w("")
    if isinstance(gate_a4, dict):
        w(f"**Shadow eligible**: {gate_a4.get('shadow_eligible', False)}")
        w(f"**Active eligible**: {gate_a4.get('active_eligible', False)}")
        blocking = gate_a4.get("blocking_reasons", [])
        if blocking:
            w("")
            w("**Blocking reasons**:")
            for r in blocking:
                w(f"- {r}")
    w("")
    # Legacy gate A status — conservative.
    w(f"**Legacy Gate A status**: {legacy_gate_a}")
    w("")
    if legacy_gate_a == "PASS":
        w(
            "Gate A passed (Gate A2 AND Gate A3 both passed). "
            "The candidate is eligible for promotion consideration."
        )
    elif legacy_gate_a == "BLOCKED":
        w(
            "Gate A is BLOCKED. One or more gates could not be evaluated "
            "due to missing evidence."
        )
    else:
        w(
            "Gate A did NOT pass. Gate A2 and Gate A3 must both pass for "
            "Gate A to be considered passed."
        )
    w("")

    # --- Section 17: Limitations ---
    w("## 17. Limitations")
    w("")
    if is_synthetic:
        w(
            "- This run uses SYNTHETIC evidence. Synthetic evidence may test "
            "pipeline correctness but may NEVER satisfy a real-model causal "
            "efficacy claim."
        )
        w("- No real-model inference was performed.")
        w("- Results cannot be promoted and cannot support model-scale decisions.")
    elif is_unavailable:
        w(
            "- The original source artifacts are not available in this "
            "repository. The historical Gate A result cannot be reproduced."
        )
        w(
            "- UNVERIFIED HISTORICAL CLAIM — SOURCE ARTIFACTS NOT INCLUDED."
        )
        w("- Synthetic fixtures are maintained separately and are not used as substitutes.")
    else:
        w("- Results are specific to the model, benchmark, and configuration recorded above.")
        w("- Causal claims depend on the correctness of the counterfactual equivalence verification.")
        w("- Family-level conclusions are limited to families with sufficient support.")
        if a3_status != "PASS":
            w("- Replication robustness was not fully established.")
        if a2_status != "PASS":
            w("- Candidate causal effectiveness was not demonstrated.")
    w("")

    # --- Section 18: Exact commands to reproduce ---
    w("## 18. Exact Commands to Reproduce")
    w("")
    w("```bash")
    w("# Generate evidence (requires GPU):")
    w(f"python -m qualification.autolearn_v04.run_local_gpu_scientific \\")
    w(f"    --config <config.yaml> --output-root {config.get('output_root', 'qualification/evidence/runs')}")
    w("")
    w("# Audit evidence admissibility (Gate A0):")
    w(f"python -m qualification.autolearn_v04.v047.cli audit \\")
    w(f"    --evidence-dir <run_dir>")
    w("")
    w("# Evaluate all gates (A1/A2/A3/A4):")
    w(f"python -m qualification.autolearn_v04.v047.cli evaluate \\")
    w(f"    --evidence-dir <run_dir>")
    w("")
    w("# Generate this report:")
    w(f"python -m qualification.autolearn_v04.v047.cli report \\")
    w(f"    --evidence-dir <run_dir>")
    w("")
    w("# Verify checksums:")
    w(f"python -m qualification.autolearn_v04.v047.cli verify-checksums \\")
    w(f"    --evidence-dir <run_dir>")
    w("")
    w("# Promotion check:")
    w(f"python -m qualification.autolearn_v04.v047.cli promotion-check \\")
    w(f"    --evidence-dir <run_dir>")
    w("```")
    w("")

    # --- Section 19: Checksums ---
    w("## 19. Checksums")
    w("")
    if checksums and isinstance(checksums, dict):
        w(f"**Valid**: {checksums.get('valid', False)}")
        w(f"**Mismatches**: {len(checksums.get('mismatches', []))}")
        w(f"**Missing**: {len(checksums.get('missing', []))}")
        w(f"**Extra**: {len(checksums.get('extra', []))}")
    else:
        w("**Checksums**: No checksum verification data available.")
    w("")
    w(
        "*The CHECKSUMS.sha256 file cryptographically binds every artifact "
        "in the run directory. Any post-hoc mutation is detected by "
        "verify-checksums.*"
    )
    w("")

    return "\n".join(lines)


__all__ = ["generate_v047_report"]
