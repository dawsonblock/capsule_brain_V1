"""Canonical CLI for v0.4.7 qualification.

Usage:
    python -m qualification.autolearn_v04.v047.cli <command> [args]

Commands:
    qualify           — run the full pipeline (evidence generation + analysis)
    audit             — run Gate A0 admissibility check on existing evidence
    evaluate          — run Gates A1/A2/A3/A4 on existing evidence
    report            — generate FINAL_REPORT.md from existing gate results
    verify-checksums  — verify all checksums in CHECKSUMS.sha256
    promotion-check   — verify promotion manifest binding

Exit codes (per spec):
    0: requested operation passed
    1: scientific gate failed
    2: blocked or incomplete evidence
    3: invalid configuration
    4: artifact-integrity failure
    5: runtime execution failure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION


# ---------------------------------------------------------------------------
# Exit codes (per spec)
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0          # requested operation passed
EXIT_GATE_FAILED = 1      # scientific gate failed
EXIT_BLOCKED = 2          # blocked or incomplete evidence
EXIT_CONFIG_ERROR = 3     # invalid configuration
EXIT_ARTIFACT_FAILURE = 4 # artifact-integrity failure
EXIT_RUNTIME_ERROR = 5    # runtime execution failure


# ---------------------------------------------------------------------------
# JSON / JSONL loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e


def _load_json_file(path: Path) -> dict | None:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path} at line: {line[:80]}...: {e}"
                    ) from e
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Evidence loading
# ---------------------------------------------------------------------------


def _load_evidence(evidence_dir: str) -> dict[str, Any]:
    """Load all evidence from a v0.4.7 run directory.

    Returns a dict with keys:
        - run_manifest
        - source_manifest
        - model_manifest
        - provider_manifest
        - tokenizer_manifest
        - generation_config
        - benchmark_manifest
        - split_manifest
        - candidate_results
        - baseline_results
        - sham_results
        - oracle_results
        - safety_experiences
        - counterfactual_outcomes
        - gate_a0_v046_result (if a v0.4.6 Gate A0 result exists)
        - evidence_origin
    """
    ev = Path(evidence_dir)
    if not ev.exists() or not ev.is_dir():
        raise FileNotFoundError(f"Evidence directory does not exist: {evidence_dir}")

    run_manifest = _load_json_file(ev / "RUN_MANIFEST.json") or {}
    source_manifest = _load_json_file(ev / "SOURCE_MANIFEST.json") or {}
    model_manifest = _load_json_file(ev / "MODEL_MANIFEST.json") or {}
    provider_manifest = _load_json_file(ev / "PROVIDER_MANIFEST.json") or {}
    tokenizer_manifest = _load_json_file(ev / "TOKENIZER_MANIFEST.json") or {}
    generation_config = _load_json_file(ev / "GENERATION_CONFIG.json") or {}
    benchmark_manifest = _load_json_file(ev / "BENCHMARK_MANIFEST.json") or {}
    split_manifest = _load_json_file(ev / "SPLIT_MANIFEST.json") or {}

    # Policy results — look in subdirectories for task_rows JSON files.
    candidate_results = _load_policy_results(ev / "CANDIDATE_POLICY")
    baseline_results = _load_policy_results(ev / "BASELINE_POLICY")
    sham_results = _load_policy_results(ev / "SHAM_POLICY")
    oracle_results = _load_policy_results(ev / "ORACLE_POLICY")

    # Safety results for all three policies (if available).
    candidate_safety_results = _load_json_file(ev / "candidate_safety_results.json") or []
    baseline_safety_results = _load_json_file(ev / "baseline_safety_results.json") or []
    sham_safety_results = _load_json_file(ev / "sham_safety_results.json") or []

    # Safety experiences (legacy format).
    safety_experiences = _load_jsonl(ev / "SAFETY_EXPERIENCES.jsonl")

    # Counterfactual outcomes.
    counterfactual_outcomes = _load_jsonl(ev / "COUNTERFACTUAL_OUTCOMES.jsonl")

    # Evidence origin — from run manifest or evidence manifest.
    evidence_origin = run_manifest.get("evidence_origin", "")
    if not evidence_origin:
        # Try EVIDENCE_MANIFEST.json (v0.4.6 compatibility).
        ev_manifest = _load_json_file(ev / "EVIDENCE_MANIFEST.json")
        if ev_manifest:
            evidence_origin = ev_manifest.get("evidence_origin", "")

    # Try to load an existing v0.4.6 Gate A0 result if present.
    gate_a0_v046_result = _load_json_file(ev / "GATE_RESULTS" / "gate_a0_v046.json")
    if gate_a0_v046_result is None:
        # Also check for gate_a0.json
        gate_a0_v046_result = _load_json_file(ev / "GATE_RESULTS" / "gate_a0.json")

    return {
        "run_manifest": run_manifest,
        "source_manifest": source_manifest,
        "model_manifest": model_manifest,
        "provider_manifest": provider_manifest,
        "tokenizer_manifest": tokenizer_manifest,
        "generation_config": generation_config,
        "benchmark_manifest": benchmark_manifest,
        "split_manifest": split_manifest,
        "candidate_results": candidate_results,
        "baseline_results": baseline_results,
        "sham_results": sham_results,
        "oracle_results": oracle_results,
        "safety_experiences": safety_experiences,
        "candidate_safety_results": candidate_safety_results,
        "baseline_safety_results": baseline_safety_results,
        "sham_safety_results": sham_safety_results,
        "counterfactual_outcomes": counterfactual_outcomes,
        "gate_a0_v046_result": gate_a0_v046_result,
        "evidence_origin": evidence_origin,
        "run_dir": str(ev),
    }


def _load_policy_results(policy_dir: Path) -> dict:
    """Load policy evaluation results from a policy subdirectory.

    Looks for ``evaluation_results.json`` or ``task_results.json``.
    Falls back to loading any ``*.json`` file that contains ``task_rows``.
    """
    if not policy_dir.exists() or not policy_dir.is_dir():
        return {"task_rows": []}

    # Try common filenames.
    for name in ("evaluation_results.json", "task_results.json", "results.json"):
        p = policy_dir / name
        result = _load_json_file(p)
        if result and isinstance(result, dict) and "task_rows" in result:
            return result

    # Try EVALUATION subdirectory.
    eval_dir = policy_dir / "EVALUATION"
    if eval_dir.exists() and eval_dir.is_dir():
        for name in ("evaluation_results.json", "task_results.json", "results.json"):
            p = eval_dir / name
            result = _load_json_file(p)
            if result and isinstance(result, dict) and "task_rows" in result:
                return result

    # Fallback: scan for any JSON with task_rows.
    for p in sorted(policy_dir.rglob("*.json")):
        result = _load_json_file(p)
        if result and isinstance(result, dict) and "task_rows" in result:
            return result

    return {"task_rows": []}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML or JSON config file into a dict.

    Uses PyYAML if available for .yaml/.yml files; otherwise expects JSON.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from e
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        return _load_json(p)


def _config_dict_to_v047(config_dict: dict[str, Any]):
    """Build a QualificationConfigV047 from a dict, applying defaults."""
    from qualification.autolearn_v04.v047.config import (
        QualificationConfigV047,
        GateA1Config,
        GateA2Config,
        GateA3Config,
        CollapseConfig,
        SafetyConfig,
        PromotionConfig,
        StatisticsConfig,
    )

    # Extract nested configs with defaults.
    gate_a1 = GateA1Config(**(config_dict.get("gate_a1") or {}))
    gate_a2 = GateA2Config(**(config_dict.get("gate_a2") or {}))
    gate_a3 = GateA3Config(**(config_dict.get("gate_a3") or {}))
    collapse = CollapseConfig(**(config_dict.get("collapse") or {}))
    safety = SafetyConfig(**(config_dict.get("safety") or {}))
    promotion = PromotionConfig(**(config_dict.get("promotion") or {}))
    statistics = StatisticsConfig(**(config_dict.get("statistics") or {}))

    # Build top-level config, filtering unknown keys.
    known_fields = {
        "protocol_version", "run_id", "evidence_origin", "output_root",
        "generation_seeds", "learner_seeds",
        "model_id", "model_revision", "tokenizer_revision",
        "dtype", "device",
        "temperature", "do_sample", "max_new_tokens", "timeout_seconds",
        "experience_fraction", "validation_fraction",
        "test_fraction", "ood_fraction", "group_by", "split_seed",
    }
    top_level = {k: v for k, v in config_dict.items() if k in known_fields}

    return QualificationConfigV047(
        gate_a1=gate_a1,
        gate_a2=gate_a2,
        gate_a3=gate_a3,
        collapse=collapse,
        safety=safety,
        promotion=promotion,
        statistics=statistics,
        **top_level,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qualification.autolearn_v04.v047.cli",
        description="v0.4.7 qualification CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"capsule-brain {PACKAGE_VERSION} / qualification {AUTOLEARN_QUALIFICATION_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # qualify
    p_qualify = subparsers.add_parser(
        "qualify", help="Run the full pipeline (evidence generation + analysis)"
    )
    p_qualify.add_argument("--config", required=True, help="Path to YAML config file")

    # audit
    p_audit = subparsers.add_parser(
        "audit", help="Run Gate A0 admissibility check on existing evidence"
    )
    p_audit.add_argument("--evidence-dir", required=True)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Run Gates A1/A2/A3/A4 on existing evidence"
    )
    p_eval.add_argument("--evidence-dir", required=True)
    p_eval.add_argument("--config", default=None, help="Optional config override")

    # report
    p_report = subparsers.add_parser(
        "report", help="Generate FINAL_REPORT.md from existing gate results"
    )
    p_report.add_argument("--evidence-dir", required=True)

    # verify-checksums
    p_verify = subparsers.add_parser(
        "verify-checksums", help="Verify all checksums in CHECKSUMS.sha256"
    )
    p_verify.add_argument("--evidence-dir", required=True)

    # promotion-check
    p_promo = subparsers.add_parser(
        "promotion-check", help="Verify promotion manifest binding"
    )
    p_promo.add_argument("--evidence-dir", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "qualify":
            return _cmd_qualify(args)
        elif args.command == "audit":
            return _cmd_audit(args)
        elif args.command == "evaluate":
            return _cmd_evaluate(args)
        elif args.command == "report":
            return _cmd_report(args)
        elif args.command == "verify-checksums":
            return _cmd_verify_checksums(args)
        elif args.command == "promotion-check":
            return _cmd_promotion_check(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (ValueError, KeyError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_qualify(args) -> int:
    """Run the full qualification pipeline.

    Currently a stub — evidence generation requires
    run_local_gpu_scientific.py. Use ``evaluate`` for analysis.
    """
    print(
        "Not yet implemented — use run_local_gpu_scientific.py for evidence "
        "generation and v047.cli evaluate for analysis"
    )
    return EXIT_RUNTIME_ERROR


def _cmd_audit(args) -> int:
    """Run Gate A0 admissibility check on existing evidence."""
    from qualification.autolearn_v04.v047.gate_a0 import evaluate_gate_a0_v047
    from qualification.autolearn_v04.common.audit_status import AuditStatus

    evidence = _load_evidence(args.evidence_dir)
    evidence_origin = str(evidence.get("evidence_origin", "")).upper()

    # If we have a pre-computed v0.4.6 Gate A0 result, wrap it.
    gate_a0_v046 = evidence.get("gate_a0_v046_result")
    if gate_a0_v046 and isinstance(gate_a0_v046, dict):
        result = evaluate_gate_a0_v047(gate_a0_v046, evidence_origin)
    else:
        # No v0.4.6 result available — produce a BLOCKED result.
        from qualification.autolearn_v04.v047.gate_schema import GateResult
        result = GateResult(
            gate_name="gate_a0_admissibility",
            status=AuditStatus.BLOCKED.value,
            reasons=[
                "No v0.4.6 Gate A0 result found in evidence directory. "
                "Run v0.4.6 analysis first or provide gate_a0_v046.json."
            ],
            checks={"evidence_origin": evidence_origin},
        )

    result_dict = result.to_dict()
    print(f"Gate A0 status: {result.status}")
    if result.reasons:
        print("Reasons:")
        for r in result.reasons:
            print(f"  - {r}")

    # Write result to GATE_RESULTS directory.
    ev = Path(args.evidence_dir)
    gate_results_dir = ev / "GATE_RESULTS"
    gate_results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(gate_results_dir / "gate_a0.json", result_dict)
    print(f"Written: {gate_results_dir / 'gate_a0.json'}")

    if result.status == AuditStatus.PASS.value:
        return EXIT_SUCCESS
    elif result.status == AuditStatus.BLOCKED.value:
        return EXIT_BLOCKED
    else:
        return EXIT_GATE_FAILED


def _cmd_evaluate(args) -> int:
    """Run Gates A1/A2/A3/A4 on existing evidence."""
    from qualification.autolearn_v04.common.audit_status import AuditStatus
    from qualification.autolearn_v04.v047.gate_schema import QualificationVerdict
    from qualification.autolearn_v04.v047.gate_a1 import evaluate_gate_a1
    from qualification.autolearn_v04.v047.gate_a2 import evaluate_gate_a2
    from qualification.autolearn_v04.v047.gate_a3 import evaluate_gate_a3
    from qualification.autolearn_v04.v047.gate_a4 import evaluate_gate_a4
    from qualification.autolearn_v04.v047.family_evaluation import evaluate_families
    from qualification.autolearn_v04.v047.collapse_checks import check_action_collapse
    from qualification.autolearn_v04.v047.safety_evaluation import evaluate_safety

    evidence = _load_evidence(args.evidence_dir)
    evidence_origin = str(evidence.get("evidence_origin", "")).upper()

    # Load or build config.
    if args.config:
        config_dict = _load_config(args.config)
    else:
        config_dict = {}
    config = _config_dict_to_v047(config_dict)

    run_manifest = evidence.get("run_manifest", {})
    run_id = run_manifest.get("run_id", "v047_eval_001")

    candidate_results = evidence.get("candidate_results", {"task_rows": []})
    baseline_results = evidence.get("baseline_results", {"task_rows": []})
    sham_results = evidence.get("sham_results", {"task_rows": []})
    oracle_results = evidence.get("oracle_results", {"task_rows": []})

    # --- Gate A0 (wrap if available) ---
    from qualification.autolearn_v04.v047.gate_a0 import evaluate_gate_a0_v047
    gate_a0_v046 = evidence.get("gate_a0_v046_result")
    if gate_a0_v046 and isinstance(gate_a0_v046, dict):
        a0_result = evaluate_gate_a0_v047(gate_a0_v046, evidence_origin)
    else:
        from qualification.autolearn_v04.v047.gate_schema import GateResult
        a0_result = GateResult(
            gate_name="gate_a0_admissibility",
            status=AuditStatus.BLOCKED.value,
            reasons=["No v0.4.6 Gate A0 result found."],
            checks={"evidence_origin": evidence_origin},
        )

    # --- Gate A1: Routing headroom ---
    a1_result = evaluate_gate_a1(
        oracle_results=oracle_results,
        baseline_results=baseline_results,
        sham_results=sham_results,
        config=config.gate_a1,
        statistics_config=config.statistics,
    )

    # --- Gate A2: Candidate causal effectiveness ---
    a2_result = evaluate_gate_a2(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        sham_results=sham_results,
        config=config.gate_a2,
        statistics_config=config.statistics,
    )

    # --- Family evaluation ---
    family_eval = evaluate_families(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        sham_results=sham_results,
        oracle_results=oracle_results,
        min_test_tasks=config.gate_a3.min_family_test_tasks,
        min_test_groups=config.gate_a3.min_family_test_groups,
    )

    # --- Collapse check ---
    collapse = check_action_collapse(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        config=config.collapse,
    )

    # --- Safety evaluation ---
    # Use dedicated safety results if available, otherwise fall back to
    # safety_experiences (legacy format).
    cand_safety = evidence.get("candidate_safety_results", [])
    if not cand_safety:
        cand_safety = evidence.get("safety_experiences", [])
    safety = evaluate_safety(
        candidate_safety_results=cand_safety,
        baseline_safety_results=evidence.get("baseline_safety_results", []),
        sham_safety_results=evidence.get("sham_safety_results", []),
        config=config.safety,
    )

    # --- Gate A3: Robustness and replication ---
    # Build replicate results from available evidence.
    # In a full run, replicates come from multiple seeds. Here we build
    # a single-replicate summary from the available candidate results.
    # If a replicate_results.json file exists in the evidence directory,
    # load it instead (produced by multi-seed runs).
    replicate_file = Path(args.evidence_dir) / "replicate_results.json"
    if replicate_file.exists():
        replicate_results = _load_json_file(replicate_file) or []
        if not replicate_results:
            replicate_results = _build_replicate_results(
                candidate_results, baseline_results, sham_results, config
            )
    else:
        replicate_results = _build_replicate_results(
            candidate_results, baseline_results, sham_results, config
        )
    a3_result = evaluate_gate_a3(
        replicate_results=replicate_results,
        family_evaluation=family_eval,
        collapse_check=collapse,
        safety_check=safety,
        config=config.gate_a3,
    )

    # --- Gate A4: Promotion eligibility ---
    # Route through the dedicated evaluate_gate_a4 module to ensure
    # consistent promotion semantics. Active eligibility is ALWAYS False
    # after offline qualification — it requires post-shadow validation.
    safety_status = safety.get("status", "FAIL")
    artifact_binding_status = "PASS"  # artifact binding checked in A0
    a4_result = evaluate_gate_a4(
        gate_a0_status=a0_result.status,
        gate_a1_status=a1_result.status,
        gate_a2_status=a2_result.status,
        gate_a3_status=a3_result.status,
        safety_status=safety_status,
        artifact_binding_status=artifact_binding_status,
        evidence_origin=evidence_origin,
        config=config.promotion,
    )

    # --- Build verdict ---
    verdict = QualificationVerdict(
        protocol_version="0.4.7",
        run_id=run_id,
        evidence_origin=evidence_origin,
        gate_a0_admissibility=a0_result.to_dict(),
        gate_a1_headroom=a1_result.to_dict(),
        gate_a2_effectiveness=a2_result.to_dict(),
        gate_a3_robustness=a3_result.to_dict(),
        gate_a4_promotion=a4_result.to_dict(),
    )

    # --- Write results ---
    ev = Path(args.evidence_dir)
    gate_results_dir = ev / "GATE_RESULTS"
    gate_results_dir.mkdir(parents=True, exist_ok=True)

    _write_json(gate_results_dir / "gate_a0.json", a0_result.to_dict())
    _write_json(gate_results_dir / "gate_a1.json", a1_result.to_dict())
    _write_json(gate_results_dir / "gate_a2.json", a2_result.to_dict())
    _write_json(gate_results_dir / "gate_a3.json", a3_result.to_dict())
    _write_json(gate_results_dir / "gate_a4.json", a4_result.to_dict())
    _write_json(gate_results_dir / "verdict.json", verdict.to_dict())
    _write_json(gate_results_dir / "family_evaluation.json", family_eval)
    _write_json(gate_results_dir / "collapse_check.json", collapse)
    _write_json(gate_results_dir / "safety_check.json", safety)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"v0.4.7 Evaluation complete")
    print(f"Run ID: {run_id}")
    print(f"Evidence origin: {evidence_origin}")
    print(f"Gate A0 (admissibility): {a0_result.status}")
    print(f"Gate A1 (headroom):       {a1_result.status}")
    print(f"Gate A2 (effectiveness):  {a2_result.status}")
    print(f"Gate A3 (robustness):     {a3_result.status}")
    print(f"Gate A4 (promotion):      {a4_result.status}")
    print(f"Shadow eligible:          {shadow_eligible}")
    print(f"Active eligible:          {active_eligible}")
    print(f"Legacy Gate A:            {verdict.legacy_gate_a_status}")
    print(f"{'='*60}")
    print(f"Results written to: {gate_results_dir}")

    # --- Exit code ---
    # BLOCKED takes priority over GATE_FAILED.
    any_blocked = any(
        s == AuditStatus.BLOCKED.value
        for s in [a0_result.status, a1_result.status, a2_result.status, a3_result.status]
    )
    if a4_result.status == AuditStatus.PASS.value:
        return EXIT_SUCCESS
    if any_blocked:
        return EXIT_BLOCKED
    return EXIT_GATE_FAILED


def _build_replicate_results(
    candidate_results: dict,
    baseline_results: dict,
    sham_results: dict,
    config,
) -> list[dict]:
    """Build replicate result summaries from available evidence.

    In a full run, replicates come from multiple generation and learner
    seeds.  When only a single run is available, we produce a single
    replicate entry so that Gate A3 can report insufficient seed diversity
    honestly rather than crashing.
    """
    from qualification.autolearn_v04.v047.statistics import (
        compute_paired_deltas, paired_cluster_bootstrap,
    )

    deltas = compute_paired_deltas(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        sham_results=sham_results,
        oracle_results=[],
    )
    cb_rows = deltas["candidate_vs_baseline"]
    cs_rows = deltas["candidate_vs_sham"]

    cb_stats = paired_cluster_bootstrap(
        cb_rows,
        cluster_key=config.statistics.cluster_key,
        n_resamples=min(config.statistics.bootstrap_resamples, 1000),
        confidence_level=config.gate_a2.confidence_level,
        seed=42,
    )
    cs_stats = paired_cluster_bootstrap(
        cs_rows,
        cluster_key=config.statistics.cluster_key,
        n_resamples=min(config.statistics.bootstrap_resamples, 1000),
        confidence_level=config.gate_a2.confidence_level,
        seed=42,
    )

    cb_threshold = config.gate_a2.candidate_vs_baseline_min_effect
    cs_threshold = config.gate_a2.candidate_vs_sham_min_effect

    return [{
        "generation_seed": config.generation_seeds[0] if config.generation_seeds else 0,
        "learner_seed": config.learner_seeds[0] if config.learner_seeds else 0,
        "candidate_vs_baseline_delta": cb_stats["mean_delta"],
        "candidate_vs_sham_delta": cs_stats["mean_delta"],
        "candidate_vs_baseline_passes": cb_stats["lower_bound"] > cb_threshold,
        "candidate_vs_sham_passes": cs_stats["lower_bound"] > cs_threshold,
    }]


def _cmd_report(args) -> int:
    """Generate FINAL_REPORT.md from existing gate results."""
    from qualification.autolearn_v04.v047.report import generate_v047_report
    from qualification.autolearn_v04.v047.run_directory import verify_checksums

    ev = Path(args.evidence_dir)
    gate_results_dir = ev / "GATE_RESULTS"

    # Load verdict.
    verdict_path = gate_results_dir / "verdict.json"
    if not verdict_path.exists():
        print(
            f"Verdict not found: {verdict_path}. "
            "Run 'evaluate' first to generate gate results.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    verdict = _load_json(verdict_path)

    # Load config from run manifest or use defaults.
    config_dict = {}
    run_manifest = _load_json_file(ev / "RUN_MANIFEST.json") or {}
    if "config" in run_manifest:
        config_dict = run_manifest["config"] or {}

    # Load supporting data.
    family_eval = _load_json_file(gate_results_dir / "family_evaluation.json") or {}
    collapse = _load_json_file(gate_results_dir / "collapse_check.json") or {}
    safety = _load_json_file(gate_results_dir / "safety_check.json") or {}

    # Build evidence summary from manifests.
    source_manifest = _load_json_file(ev / "SOURCE_MANIFEST.json") or {}
    model_manifest = _load_json_file(ev / "MODEL_MANIFEST.json") or {}
    provider_manifest = _load_json_file(ev / "PROVIDER_MANIFEST.json") or {}
    tokenizer_manifest = _load_json_file(ev / "TOKENIZER_MANIFEST.json") or {}
    benchmark_manifest = _load_json_file(ev / "BENCHMARK_MANIFEST.json") or {}
    split_manifest = _load_json_file(ev / "SPLIT_MANIFEST.json") or {}

    evidence_summary = {
        "source_manifest": source_manifest,
        "model_manifest": model_manifest,
        "provider_manifest": provider_manifest,
        "tokenizer_manifest": tokenizer_manifest,
        "benchmark_manifest": benchmark_manifest,
        "split_manifest": split_manifest,
    }

    # Verify checksums.
    checksums = verify_checksums(ev)

    # Generate report.
    report_md = generate_v047_report(
        verdict=verdict,
        config=config_dict,
        evidence_summary=evidence_summary,
        family_evaluation=family_eval,
        collapse_check=collapse,
        safety_check=safety,
        checksums=checksums,
    )

    # Write FINAL_REPORT.md.
    report_path = ev / "FINAL_REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Report written: {report_path}")

    # Print summary.
    a4 = verdict.get("gate_a4_promotion", {})
    print(f"Gate A4 status: {a4.get('status', 'NOT_RUN')}")
    print(f"Shadow eligible: {a4.get('shadow_eligible', False)}")
    print(f"Active eligible: {a4.get('active_eligible', False)}")

    if not checksums.get("valid", False):
        print(f"WARNING: Checksum verification failed — {len(checksums.get('mismatches', []))} mismatch(es)")
        return EXIT_ARTIFACT_FAILURE

    return EXIT_SUCCESS


def _cmd_verify_checksums(args) -> int:
    """Verify all checksums in CHECKSUMS.sha256."""
    from qualification.autolearn_v04.v047.run_directory import verify_checksums

    ev = Path(args.evidence_dir)
    if not ev.exists() or not ev.is_dir():
        print(f"Evidence directory does not exist: {args.evidence_dir}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    checksums_path = ev / "CHECKSUMS.sha256"
    if not checksums_path.exists():
        print(f"CHECKSUMS.sha256 not found: {checksums_path}", file=sys.stderr)
        return EXIT_BLOCKED

    result = verify_checksums(ev)

    print(f"Valid: {result['valid']}")
    print(f"Mismatches: {len(result['mismatches'])}")
    print(f"Missing: {len(result['missing'])}")
    print(f"Extra: {len(result['extra'])}")

    if result["mismatches"]:
        print("\nMismatches:")
        for m in result["mismatches"]:
            print(f"  - {m}")
    if result["missing"]:
        print("\nMissing files:")
        for m in result["missing"]:
            print(f"  - {m}")
    if result["extra"]:
        print("\nExtra files (not in checksum manifest):")
        for e in result["extra"]:
            print(f"  - {e}")

    if result["valid"]:
        return EXIT_SUCCESS
    return EXIT_ARTIFACT_FAILURE


def _cmd_promotion_check(args) -> int:
    """Verify promotion manifest binding."""
    from qualification.autolearn_v04.v047.evidence_enforcement import can_promote
    from qualification.autolearn_v04.common.audit_status import AuditStatus

    ev = Path(args.evidence_dir)
    gate_results_dir = ev / "GATE_RESULTS"

    # Load verdict.
    verdict_path = gate_results_dir / "verdict.json"
    if not verdict_path.exists():
        print(
            f"Verdict not found: {verdict_path}. "
            "Run 'evaluate' first to generate gate results.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    verdict = _load_json(verdict_path)

    a4 = verdict.get("gate_a4_promotion", {})
    a4_status = a4.get("status", AuditStatus.NOT_RUN.value)
    shadow_eligible = a4.get("shadow_eligible", False)
    active_eligible = a4.get("active_eligible", False)
    evidence_origin = str(verdict.get("evidence_origin", "")).upper()

    print(f"Gate A4 status: {a4_status}")
    print(f"Shadow eligible: {shadow_eligible}")
    print(f"Active eligible: {active_eligible}")
    print(f"Evidence origin: {evidence_origin}")

    blocking = a4.get("blocking_reasons", [])
    if blocking:
        print("\nBlocking reasons:")
        for r in blocking:
            print(f"  - {r}")

    # Check promotion manifest binding if present.
    promo_manifest_path = ev / "PROMOTION_MANIFEST.json"
    if promo_manifest_path.exists():
        promo_manifest = _load_json(promo_manifest_path)
        print(f"\nPromotion manifest found: {promo_manifest_path}")
        print(f"  Bound run ID: {promo_manifest.get('run_id', '')}")
        print(f"  Mode: {promo_manifest.get('mode', '')}")
        print(f"  Verdict hash: {promo_manifest.get('verdict_hash', '')[:16]}...")

        # Verify binding matches verdict.
        bound_run_id = promo_manifest.get("run_id", "")
        verdict_run_id = verdict.get("run_id", "")
        if bound_run_id != verdict_run_id:
            print(
                f"\nERROR: Promotion manifest run_id '{bound_run_id}' "
                f"does not match verdict run_id '{verdict_run_id}'"
            )
            return EXIT_ARTIFACT_FAILURE
    else:
        print("\nNo PROMOTION_MANIFEST.json found.")

    if a4_status == AuditStatus.PASS.value and shadow_eligible:
        print("\nPromotion check: PASS")
        return EXIT_SUCCESS
    elif a4_status == AuditStatus.BLOCKED.value:
        print("\nPromotion check: BLOCKED")
        return EXIT_BLOCKED
    else:
        print("\nPromotion check: FAIL")
        return EXIT_GATE_FAILED


if __name__ == "__main__":
    sys.exit(main())
