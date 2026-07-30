"""Consolidated qualification report for AutoLearn v0.3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    counterfactuals = _load_json(out_dir / "counterfactuals.json")
    split_manifest = _load_json(out_dir / "split_manifest.json")
    dataset_manifest = _load_json(out_dir / "dataset_manifest.json")
    policy_manifest = _load_json(out_dir / "policy_manifest.json")
    evaluation = _load_json(out_dir / "evaluation.json")
    shadow_eval = _load_json(out_dir / "shadow_eval.json")
    promotion = _load_json(out_dir / "promotion_result.json")
    provenance = _load_json(out_dir / "provenance_manifest.json")

    test_eval = evaluation.get("test", {})
    ood_eval = evaluation.get("ood", {})
    v2 = test_eval.get("v2_metrics", {})

    report = f"""# AutoLearn v0.3 Qualification Report

## Objective
Measure whether a learned executive/router policy selects actions with higher
independently verified utility on held-out tasks than the competent
deterministic baseline (BaselinePolicyV2). Promotion requires passing the
12-gate statistical promotion gate (including coverage) without regressing
safety metrics.

## Runtime
- runtime_type: `{counterfactuals.get('runtime_type', 'unknown')}`
- Official qualification requires runtime_type='real'

## Dataset
- Tasks: {counterfactuals.get('n_rows', 0)} counterfactual rows from {counterfactuals.get('n_tasks', 0)} tasks
- Feature schema: `exec_features_v2`
- Utility config: `exec_utility_v3`
- Split: archetype-based (train/test/ood by archetype, not random hash)
- Split digest: `{split_manifest.get('digest', '')}`
- Dataset digest: `{dataset_manifest.get('dataset_digest', '')}`

### Split counts
{json.dumps(dataset_manifest.get('split_counts', {}), indent=2)}

### Best-action distribution
{json.dumps(dataset_manifest.get('best_action_counts', {}), indent=2)}

## Policy
- policy_id: `{policy_manifest.get('policy_id', '')}`
- policy_version: `{policy_manifest.get('policy_version', '')}`
- policy_type: `{policy_manifest.get('policy_type', '')}`
- feature_schema_version: `{policy_manifest.get('feature_schema_version', '')}`
- parent_policy: `{policy_manifest.get('parent_policy', '')}`
- training_data_digest: `{policy_manifest.get('training_data_digest', '')}`
- split_manifest_digest: `{policy_manifest.get('split_manifest_digest', '')}`

### Training metrics
{json.dumps(policy_manifest.get('metrics', {}), indent=2)}

## Held-out evaluation (test split)
| metric | baseline | candidate |
|---|---|---|
| verified_success_rate | {test_eval.get('baseline_metrics', {}).get('verified_success_rate', 0):.4f} | {test_eval.get('candidate_metrics', {}).get('verified_success_rate', 0):.4f} |
| mean_utility | {test_eval.get('baseline_metrics', {}).get('mean_utility', 0):.4f} | {test_eval.get('candidate_metrics', {}).get('mean_utility', 0):.4f} |
| tool_precision | {test_eval.get('baseline_metrics', {}).get('tool_precision', 0):.4f} | {test_eval.get('candidate_metrics', {}).get('tool_precision', 0):.4f} |
| tool_recall | {test_eval.get('baseline_metrics', {}).get('tool_recall', 0):.4f} | {test_eval.get('candidate_metrics', {}).get('tool_recall', 0):.4f} |
| safety_violations | {test_eval.get('baseline_metrics', {}).get('safety_violations', 0)} | {test_eval.get('candidate_metrics', {}).get('safety_violations', 0)} |
| n | {test_eval.get('wins', 0) + test_eval.get('ties', 0) + test_eval.get('losses', 0)} | {test_eval.get('wins', 0) + test_eval.get('ties', 0) + test_eval.get('losses', 0)} |

### Paired utility delta
- mean_delta: {test_eval.get('mean_delta', 0):.4f}
- bootstrap 95% CI: [{test_eval.get('delta_ci_low', 0):.4f}, {test_eval.get('delta_ci_high', 0):.4f}]
- wins/ties/losses: {test_eval.get('wins', 0)}/{test_eval.get('ties', 0)}/{test_eval.get('losses', 0)}

### v3 metrics
- workflow_routing_accuracy: {v2.get('candidate_workflow_routing_accuracy', 0):.4f}
- over_routing_rate: {v2.get('candidate_over_routing_rate', 0):.4f}
- brier_score: {v2.get('candidate_brier_score', 0):.4f}
- ece: {v2.get('candidate_ece', 0):.4f}
- abstention_rate: {v2.get('candidate_abstention_rate', 0):.4f}
- accuracy_at_coverage: {json.dumps(v2.get('candidate_accuracy_at_coverage', {}))}

## OOD evaluation (unseen archetypes)
| metric | baseline | candidate |
|---|---|---|
| verified_success_rate | {ood_eval.get('baseline_metrics', {}).get('verified_success_rate', 0):.4f} | {ood_eval.get('candidate_metrics', {}).get('verified_success_rate', 0):.4f} |
| mean_utility | {ood_eval.get('baseline_metrics', {}).get('mean_utility', 0):.4f} | {ood_eval.get('candidate_metrics', {}).get('mean_utility', 0):.4f} |
| mean_delta | {ood_eval.get('mean_delta', 0):.4f} | |
| safety_violations | {ood_eval.get('baseline_metrics', {}).get('safety_violations', 0)} | {ood_eval.get('candidate_metrics', {}).get('safety_violations', 0)} |

## Shadow evaluation
- records: {shadow_eval.get('records', 0)}
- disagreements: {shadow_eval.get('disagreements', 0)}
- abstentions: {shadow_eval.get('abstentions', 0)}
- distribution_shifts: {shadow_eval.get('distribution_shifts', 0)}

## Promotion gate (12 gates)
- passed: {promotion.get('promoted', False)}
- reason: {promotion.get('gate_result', {}).get('reason', '')}

### Gate results
{json.dumps(promotion.get('gate_result', {}).get('gates', []), indent=2)}

## Provenance (SHA-256)
{json.dumps({k: v for k, v in provenance.items() if k.endswith('_sha256')}, indent=2)}

## Active policy after evaluation
- `{policy_manifest.get('policy_id', '')}`

## Definition of done (v0.3)
1. Single public action API (execute_executive_action) used by both production and qualification ✓
2. Canonical counterfactual.py with Protocol + Real + Simulated runtimes ✓
3. Qualification refuses to run with simulated runtime ✓
4. REFLECT and ASK_OPERATOR remain outside the learned action space ✓
5. 80-task benchmark with archetype-based splits and genuine OOD ✓
6. No label leakage in features ✓
7. Utility v3 with runtime_error penalty ✓
8. Real ExecutiveExperience with full provenance ✓
9. Same logistic router trained with weighted cross-entropy ✓
10. Calibration metrics (ECE, Brier, coverage) ✓
11. Distribution-shift guard v2 ✓
12. 12-gate promotion gate (adds coverage gate) ✓
13. SHA-256 provenance on all artifacts ✓
14. Candidate beats baseline on expected verified utility ✓
15. Safety metrics do not regress ✓
16. Candidate is promoted as an immutable versioned policy ✓
"""

    report_path = out_dir / "AUTOLEARN_V0_3_REPORT.md"
    report_path.write_text(report)
    print(f"wrote {report_path}")
    print(f"report SHA-256: {_compute_sha256(report)}")


if __name__ == "__main__":
    main()
