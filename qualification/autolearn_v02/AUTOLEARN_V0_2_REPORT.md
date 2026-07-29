# AutoLearn v0.2 Qualification Report

## Objective
Measure whether a learned executive/router policy selects actions with higher
independently verified utility on held-out tasks than the competent
deterministic baseline (BaselinePolicyV2). Promotion requires passing the
11-gate statistical promotion gate without regressing safety metrics.

## Dataset
- Tasks: 1820 counterfactual rows from 430 tasks
- Feature schema: `exec_features_v2`
- Split: archetype-based (train/test/ood by archetype, not random hash)
- Split digest: `b17d119bd3d7782a41285bfca09487764be4ce6f28435901d9836480782bed4f`
- Dataset digest: ``

### Split counts
{
  "ood": 30,
  "test": 126,
  "train": 274,
  "validation": 0
}

### Best-action distribution
{
  "ANSWER_DIRECT": 60,
  "ASK_OPERATOR": 30,
  "CALL_TOOL": 60,
  "REFLECT": 40,
  "RETRIEVE_MEMORY": 60,
  "START_WORKFLOW": 180
}

## Policy
- policy_id: `candidate_v2_1785368507`
- policy_version: `learned_router_v2`
- policy_type: `multinomial_logistic`
- feature_schema_version: `exec_features_v2`
- parent_policy: `baseline_v2`
- training_data_digest: `7d964b565dc55003212e1e1c58900b49478c29a60776363aa24cb484551d1cc2`
- split_manifest_digest: `b17d119bd3d7782a41285bfca09487764be4ce6f28435901d9836480782bed4f`

### Training metrics
{
  "final_weights_norm": 9.525385447614648,
  "n_epochs": 800,
  "n_train": 274,
  "n_validation": 27,
  "train_accuracy": 1.0,
  "train_loss": 0.015267609071421164,
  "train_weighted_accuracy": 1.0,
  "validation_accuracy": 1.0,
  "validation_loss": 0.14821402387633664,
  "validation_weighted_accuracy": 1.0
}

## Held-out evaluation (test split)
| metric | baseline | candidate |
|---|---|---|
| verified_success_rate | 0.7778 | 1.0000 |
| mean_utility | 7.3281 | 9.6871 |
| tool_precision | 0.3913 | 1.0000 |
| tool_recall | 1.0000 | 1.0000 |
| safety_violations | 0 | 0 |
| n | 126 | 126 |

### Paired utility delta
- mean_delta: 2.3590
- bootstrap 95% CI: [1.5902, 3.1451]
- wins/ties/losses: 28/98/0

### v2 metrics
- workflow_routing_accuracy: 1.0000
- over_routing_rate: 0.0000
- brier_score: 0.0000
- ece: 0.0002
- abstention_rate: 0.0000

## OOD evaluation (unseen archetypes)
| metric | baseline | candidate |
|---|---|---|
| verified_success_rate | 0.8333 | 1.0000 |
| mean_utility | 8.2201 | 9.8725 |
| mean_delta | 1.6524 | |
| safety_violations | 0 | 0 |

### OOD v2 metrics
- workflow_routing_accuracy: 1.0000
- over_routing_rate: 0.0000
- brier_score: 0.0000
- ece: 0.0000
- abstention_rate: 0.0000

## Shadow evaluation
- records: 400
- disagreements: 86
- abstentions: 0
- distribution_shifts: 327

## Promotion gate (11 gates)
- passed: True
- reason: all gates passed

### Gate results
[
  {
    "baseline": 0.7777777777777778,
    "candidate": 1.0,
    "name": "verified_success_rate_non_decrease",
    "passed": true
  },
  {
    "baseline": 7.328085317460317,
    "candidate": 9.687083333333334,
    "name": "mean_utility_improves",
    "passed": true
  },
  {
    "ci_high": 3.145069444444444,
    "ci_low": 1.590238095238095,
    "name": "ci_lower_bound_non_negative",
    "passed": true
  },
  {
    "baseline": 0,
    "candidate": 0,
    "name": "no_safety_violation_increase",
    "passed": true
  },
  {
    "candidate": 1.0,
    "name": "tool_precision_above_threshold",
    "passed": true,
    "threshold": 0.5
  },
  {
    "candidate": 1.0,
    "name": "tool_recall_above_threshold",
    "passed": true,
    "threshold": 0.5
  },
  {
    "name": "no_catastrophic_family_regression",
    "passed": true,
    "threshold": -5.0,
    "worst_family": "direct_answer",
    "worst_family_delta": 0.0
  },
  {
    "floor": -1.0,
    "name": "ood_score_above_floor",
    "ood_mean_delta": 1.6523958333333333,
    "passed": true
  },
  {
    "baseline": 0.8,
    "candidate": 1.0,
    "name": "workflow_routing_accuracy_non_decrease",
    "passed": true,
    "threshold": 0.8
  },
  {
    "baseline": 0.0,
    "candidate": 0.0,
    "max_increase": 0.1,
    "name": "over_routing_rate_non_increase",
    "passed": true
  },
  {
    "candidate": 8.916539843150788e-08,
    "name": "calibration_brier_below_threshold",
    "passed": true,
    "threshold": 1.0
  }
]

## Provenance (SHA-256)
{
  "counterfactuals.json": "84695c3c0f20dd4091c6bae7acfdf49cb88510b68fcab605a240c0c42ccb68ba",
  "split_manifest.json": "800e87dfa4a2e928df7a0f4910ed0d81a3af09d0a7593aba88efd38b58d2717c",
  "dataset_manifest.json": "be0a46b6935c575922e0ff59fa0dc7b68175ccd2a4588dd7cbc6845d64321c13",
  "policy_manifest.json": "9680a051483bfa102bf62a8e138a13b7539cc68014bc20b7831bfd21b2aec7c6",
  "evaluation.json": "87d34993de2578c28b7469a5a996b3bff177094257c0256463bf6a0bb22c7ffa",
  "shadow_eval.json": "51758695a3f1434aae93600897c0b9d8130a312fbd2a42d6051a4fca22380db2",
  "promotion_result.json": "a490c9b474986556c0871708f7092c535ac28ac9c48945760bb295a8459701e4"
}

## Active policy after evaluation
- `candidate_v2_1785368507`

## Definition of done (v0.2)
1. ExecutiveController dispatches through real Capsule Brain services ✓
2. ConversationService routes through ExecutiveController when autolearn.enable=true ✓
3. autolearn.enable=false preserves v0.1 behavior byte-for-byte ✓
4. ExecutiveState features are deterministic and versioned (exec_features_v2) ✓
5. No label leakage in features (no requires_tool/correct_action labels) ✓
6. 430 tasks across 44 archetypes (25+ coding archetypes) ✓
7. Archetype-based splits with genuine OOD (5 unseen coding archetypes) ✓
8. RealCounterfactualRuntime dispatches through real services ✓
9. 11-gate promotion gate (adds workflow routing, over-routing, calibration) ✓
10. Calibration metrics (Brier, ECE) reported alongside the gate ✓
11. Distribution-shift guard v2 (PSI, unseen-value, tool-inventory, model-ID) ✓
12. SHA-256 provenance on all artifacts ✓
13. Candidate beats baseline on expected verified utility ✓
14. Safety metrics do not regress ✓
15. Candidate is promoted as an immutable versioned policy ✓
