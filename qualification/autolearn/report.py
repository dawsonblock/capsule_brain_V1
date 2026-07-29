"""Generate the AutoLearn v0.1 report and autolearn_results.json.

Reads all the artifacts produced by the harness and writes:
- AUTOLEARN_V0_1_REPORT.md
- autolearn_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    root = out_dir.parent.parent

    def _load(name: str) -> dict:
        p = out_dir / name
        return json.loads(p.read_text()) if p.exists() else {}

    counterfactuals = _load("counterfactuals.json")
    split_manifest = _load("split_manifest.json")
    dataset_manifest = _load("dataset_manifest.json")
    policy_manifest = _load("policy_manifest.json")
    evaluation = _load("evaluation.json")
    shadow = _load("shadow_eval.json")
    promotion = _load("promotion_result.json")

    test_eval = evaluation.get("test", {}) if evaluation else {}
    ood_eval = evaluation.get("ood", {}) if evaluation else {}

    results = {
        "autolearn_version": "v0.1",
        "generated_at": time.time(),
        "dataset": {
            "size": dataset_manifest.get("n_tasks", 0),
            "rows": dataset_manifest.get("n_rows", 0),
            "family_counts": dataset_manifest.get("family_counts", {}),
            "split_counts": dataset_manifest.get("split_counts", {}),
            "best_action_counts": dataset_manifest.get("best_action_counts", {}),
            "feature_schema_version": dataset_manifest.get("feature_schema_version", ""),
            "dataset_digest": dataset_manifest.get("dataset_digest", ""),
            "split_manifest_digest": dataset_manifest.get("split_manifest_digest", ""),
        },
        "policy": {
            "policy_id": policy_manifest.get("policy_id", ""),
            "policy_version": policy_manifest.get("policy_version", ""),
            "policy_type": policy_manifest.get("policy_type", ""),
            "feature_schema_version": policy_manifest.get("feature_schema_version", ""),
            "training_data_digest": policy_manifest.get("training_data_digest", ""),
            "split_manifest_digest": policy_manifest.get("split_manifest_digest", ""),
            "parent_policy": policy_manifest.get("parent_policy", ""),
            "hyperparameters": policy_manifest.get("hyperparameters", {}),
            "metrics": policy_manifest.get("metrics", {}),
        },
        "evaluation": {
            "test": test_eval,
            "ood": ood_eval,
        },
        "shadow": shadow,
        "promotion": promotion,
    }
    (out_dir / "autolearn_results.json").write_text(
        json.dumps(results, sort_keys=True, indent=2)
    )

    # Build the markdown report.
    bm = test_eval.get("baseline_metrics", {}) if test_eval else {}
    cm = test_eval.get("candidate_metrics", {}) if test_eval else {}
    md = []
    md.append("# AutoLearn v0.1 Report\n")
    md.append("## Objective\n")
    md.append(
        "Measure whether a learned executive/router policy selects actions with "
        "higher independently verified utility on held-out tasks than the "
        "deterministic rule-based baseline. Promotion requires passing a "
        "statistical promotion gate without regressing safety metrics.\n"
    )
    md.append("## Dataset\n")
    md.append(f"- Tasks: {dataset_manifest.get('n_tasks', 0)}\n")
    md.append(f"- Counterfactual rows: {dataset_manifest.get('n_rows', 0)}\n")
    md.append(f"- Feature schema: `{dataset_manifest.get('feature_schema_version', '')}`\n")
    md.append(f"- Split digest: `{dataset_manifest.get('split_manifest_digest', '')}`\n")
    md.append(f"- Dataset digest: `{dataset_manifest.get('dataset_digest', '')}`\n")
    md.append("\n### Task-family counts\n")
    for fam, n in sorted(dataset_manifest.get("family_counts", {}).items()):
        md.append(f"- {fam}: {n}\n")
    md.append("\n### Split counts\n")
    for sp, n in sorted(dataset_manifest.get("split_counts", {}).items()):
        md.append(f"- {sp}: {n}\n")
    md.append("\n### Best-action distribution (counterfactual argmax)\n")
    for a, n in sorted(dataset_manifest.get("best_action_counts", {}).items()):
        md.append(f"- {a}: {n}\n")

    md.append("\n## Policy\n")
    md.append(f"- policy_id: `{policy_manifest.get('policy_id', '')}`\n")
    md.append(f"- policy_version: `{policy_manifest.get('policy_version', '')}`\n")
    md.append(f"- policy_type: `{policy_manifest.get('policy_type', '')}`\n")
    md.append(f"- feature_schema_version: `{policy_manifest.get('feature_schema_version', '')}`\n")
    md.append(f"- parent_policy: `{policy_manifest.get('parent_policy', '')}`\n")
    md.append(f"- training_data_digest: `{policy_manifest.get('training_data_digest', '')}`\n")
    md.append(f"- split_manifest_digest: `{policy_manifest.get('split_manifest_digest', '')}`\n")
    md.append("\n### Training metrics\n")
    for k, v in sorted((policy_manifest.get("metrics", {}) or {}).items()):
        md.append(f"- {k}: {v}\n")

    md.append("\n## Held-out evaluation (test split)\n")
    md.append("| metric | baseline | candidate |\n")
    md.append("|---|---|---|\n")
    for k in [
        "verified_success_rate", "mean_utility", "median_utility",
        "tool_precision", "tool_recall", "workflow_routing_accuracy",
        "unnecessary_reflection_rate", "operator_escalation_rate",
        "median_latency", "p95_latency", "tokens_per_successful_task",
        "failure_rate", "safety_violations", "n",
    ]:
        md.append(f"| {k} | {bm.get(k, 0)} | {cm.get(k, 0)} |\n")

    md.append("\n### Paired utility delta\n")
    md.append(f"- mean_delta: {test_eval.get('mean_delta', 0)}\n")
    md.append(f"- median_delta: {test_eval.get('median_delta', 0)}\n")
    md.append(f"- bootstrap 95% CI: [{test_eval.get('delta_ci_low', 0)}, {test_eval.get('delta_ci_high', 0)}]\n")
    md.append(f"- wins/ties/losses: {test_eval.get('wins', 0)}/{test_eval.get('ties', 0)}/{test_eval.get('losses', 0)}\n")

    md.append("\n### Routing confusion matrix (baseline -> candidate)\n")
    confusion = test_eval.get("confusion", {}) if test_eval else {}
    if confusion:
        actions = sorted(confusion.keys())
        md.append("| baseline \\ candidate | " + " | ".join(actions) + " |\n")
        md.append("|" + "---|" * (len(actions) + 1) + "\n")
        for a in actions:
            row = confusion.get(a, {})
            md.append(f"| {a} | " + " | ".join(str(row.get(b, 0)) for b in actions) + " |\n")

    md.append("\n## OOD evaluation\n")
    if ood_eval:
        md.append(f"- mean_delta: {ood_eval.get('mean_delta', 0)}\n")
        md.append(f"- baseline safety_violations: {ood_eval.get('baseline_metrics', {}).get('safety_violations', 0)}\n")
        md.append(f"- candidate safety_violations: {ood_eval.get('candidate_metrics', {}).get('safety_violations', 0)}\n")
    else:
        md.append("- no OOD split\n")

    md.append("\n## Shadow evaluation\n")
    if shadow:
        stats = shadow.get("stats", {})
        md.append(f"- records: {stats.get('n', 0)}\n")
        md.append(f"- disagreements: {stats.get('disagreements', 0)}\n")
        md.append(f"- abstentions: {stats.get('abstentions', 0)}\n")
        md.append(f"- distribution_shifts: {stats.get('distribution_shifts', 0)}\n")

    md.append("\n## Promotion gate\n")
    if promotion:
        gate = promotion.get("gate_result", {})
        md.append(f"- passed: {promotion.get('promoted', False)}\n")
        md.append(f"- reason: {gate.get('reason', '')}\n")
        md.append("\n### Gate results\n")
        for g in gate.get("gates", []):
            md.append(f"- {g['name']}: passed={g['passed']}\n")

    md.append("\n## Active policy after evaluation\n")
    active = ""
    if promotion.get("promoted", False):
        active = policy_manifest.get("policy_id", "")
    else:
        active = "baseline_v1 (candidate rejected or not promoted)"
    md.append(f"- `{active}`\n")

    md.append("\n## Definition of done\n")
    dod = [
        "Capsule Brain v2.14 runtime remains unchanged except for minimal policy integration",
        "Executive experiences are recorded with verified outcomes",
        "Counterfactual action outcomes exist for the training benchmark",
        "A learned router is trained from those outcomes",
        "Candidate policy is evaluated on held-out tasks",
        "Candidate beats baseline on expected verified utility",
        "Statistical promotion gate passes",
        "Safety metrics do not regress",
        "Candidate is promoted as an immutable versioned policy",
        "Restart loads the same promoted policy",
        "Failed/incompatible policy always falls back safely",
        "Results are reproducible from manifests and hashes",
    ]
    for i, item in enumerate(dod, 1):
        md.append(f"{i}. {item}\n")

    (root / "AUTOLEARN_V0_1_REPORT.md").write_text("".join(md))
    print(f"wrote {root / 'AUTOLEARN_V0_1_REPORT.md'}")
    print(f"wrote {out_dir / 'autolearn_results.json'}")


if __name__ == "__main__":
    main()
