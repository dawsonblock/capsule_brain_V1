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
- policy_id: `candidate_v2_1785367929`
- policy_version: `learned_router_v2`
- policy_type: `multinomial_logistic`
- feature_schema_version: `exec_features_v2`
- parent_policy: `baseline_v2`
- training_data_digest: `546dc7f80d5d1d8c5c7b19be79ac6bca446b2e7c30960a1f6e045cf9af764137`
- split_manifest_digest: `b17d119bd3d7782a41285bfca09487764be4ce6f28435901d9836480782bed4f`

### Training metrics
{
  "final_weights_norm": 9.346107111581473,
  "n_epochs": 800,
  "n_train": 274,
  "n_validation": 27,
  "train_accuracy": 1.0,
  "train_loss": 0.014644685547794753,
  "train_weighted_accuracy": 1.0,
  "validation_accuracy": 1.0,
  "validation_loss": 0.1405044314262765,
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
    "candidate": 1.5447585586980076e-07,
    "name": "calibration_brier_below_threshold",
    "passed": true,
    "threshold": 1.0
  }
]

## Provenance (SHA-256)
{
  "counterfactuals.json": "05aa8773f275e67b88113b7823553d40e1d84d7c3a107b248fb6f7bc78e0df11",
  "split_manifest.json": "800e87dfa4a2e928df7a0f4910ed0d81a3af09d0a7593aba88efd38b58d2717c",
  "dataset_manifest.json": "fbca3a88ce57ecd284fae27cae830abcb53e164cc19a5b135a92ced3ce3fec36",
  "policy_manifest.json": "06a212287038e24063f880dee7a70c2e6fce8d8e75c435b621f97baab1463298",
  "evaluation.json": "b17c2ea87a6c72584d3a2224adad7cf2a8787d43214a1f745d5bfaedfcb1a49c",
  "shadow_eval.json": "c333b09c6016b6af8de871a3426c24239d713befc53c5f1836b17d13abda5a76",
  "promotion_result.json": "8c85a5efaafd7a2125e13d2229dcdaacda15721ffdc61543255d8577602b3df4"
}

## Active policy after evaluation
- `candidate_v2_1785367929`

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
