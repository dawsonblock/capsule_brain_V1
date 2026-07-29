# AutoLearn v0.1 Report
## Objective
Measure whether a learned executive/router policy selects actions with higher independently verified utility on held-out tasks than the deterministic rule-based baseline. Promotion requires passing a statistical promotion gate without regressing safety metrics.
## Dataset
- Tasks: 200
- Counterfactual rows: 860
- Feature schema: `exec_features_v1`
- Split digest: `0fccee6e74d43c3117895bfcddd535fc6433cc6dcf459d6109be16a4b40e26fc`
- Dataset digest: `3b257b702ecdc59588dbd6d846b5515fc268462fdd2c1a9dfdb3801f6c897ff2`

### Task-family counts
- coding_workflow: 40
- direct_answer: 40
- memory_required: 40
- operator_safety: 20
- reflection_req: 20
- tool_required: 40

### Split counts
- ood: 0
- test: 55
- train: 110
- validation: 35

### Best-action distribution (counterfactual argmax)
- ANSWER_DIRECT: 40
- ASK_OPERATOR: 20
- CALL_TOOL: 40
- REFLECT: 20
- RETRIEVE_MEMORY: 40
- START_WORKFLOW: 40

## Policy
- policy_id: `candidate_1785317396`
- policy_version: `learned_router_v1`
- policy_type: `multinomial_logistic`
- feature_schema_version: `exec_features_v1`
- parent_policy: `baseline_v1`
- training_data_digest: `3b257b702ecdc59588dbd6d846b5515fc268462fdd2c1a9dfdb3801f6c897ff2`
- split_manifest_digest: `0fccee6e74d43c3117895bfcddd535fc6433cc6dcf459d6109be16a4b40e26fc`

### Training metrics
- final_weights_norm: 5.767995418843203
- n_epochs: 800
- n_train: 110
- n_validation: 35
- train_accuracy: 1.0
- train_loss: 0.01824081902700862
- train_weighted_accuracy: 1.0
- validation_accuracy: 1.0
- validation_loss: 0.04960344637461444
- validation_weighted_accuracy: 1.0

## Held-out evaluation (test split)
| metric | baseline | candidate |
|---|---|---|
| verified_success_rate | 0.7884615384615384 | 1.0 |
| mean_utility | 7.66703573076923 | 9.994064576923078 |
| median_utility | 9.990995999999999 | 9.9959985 |
| tool_precision | 0.5416666666666666 | 1.0 |
| tool_recall | 1.0 | 1.0 |
| workflow_routing_accuracy | 1.0 | 1.0 |
| unnecessary_reflection_rate | 0.11538461538461539 | 0.11538461538461539 |
| operator_escalation_rate | 0.0 | 0.0 |
| median_latency | 400.0 | 400.0 |
| p95_latency | 1200.0 | 1200.0 |
| tokens_per_successful_task | 61.21951219512195 | 54.61538461538461 |
| failure_rate | 0.21153846153846154 | 0.0 |
| safety_violations | 0 | 0 |
| n | 52 | 52 |

### Paired utility delta
- mean_delta: 2.327028846153846
- median_delta: 0.0
- bootstrap 95% CI: [1.2692884615384612, 3.596317307692307]
- wins/ties/losses: 11/41/0

### Routing confusion matrix (baseline -> candidate)
| baseline \ candidate | ANSWER_DIRECT | ASK_OPERATOR | CALL_TOOL | REFLECT | RETRIEVE_MEMORY | START_WORKFLOW |
|---|---|---|---|---|---|---|
| ANSWER_DIRECT | 10 | 0 | 0 | 0 | 0 | 0 |
| ASK_OPERATOR | 0 | 0 | 0 | 0 | 0 | 0 |
| CALL_TOOL | 0 | 0 | 13 | 0 | 11 | 0 |
| REFLECT | 0 | 0 | 0 | 6 | 0 | 0 |
| RETRIEVE_MEMORY | 0 | 0 | 0 | 0 | 0 | 0 |
| START_WORKFLOW | 0 | 0 | 0 | 0 | 0 | 12 |

## OOD evaluation
- mean_delta: 7.001000000000001
- baseline safety_violations: 0
- candidate safety_violations: 0

## Shadow evaluation
- records: 90
- disagreements: 23
- abstentions: 0
- distribution_shifts: 0

## Promotion gate
- passed: True
- reason: all gates passed

### Gate results
- verified_success_rate_non_decrease: passed=True
- mean_utility_improves: passed=True
- ci_lower_bound_non_negative: passed=True
- no_safety_violation_increase: passed=True
- tool_precision_above_threshold: passed=True
- tool_recall_above_threshold: passed=True
- no_catastrophic_family_regression: passed=True
- ood_score_above_floor: passed=True

## Active policy after evaluation
- `candidate_1785317396`

## Definition of done
1. Capsule Brain v2.14 runtime remains unchanged except for minimal policy integration
2. Executive experiences are recorded with verified outcomes
3. Counterfactual action outcomes exist for the training benchmark
4. A learned router is trained from those outcomes
5. Candidate policy is evaluated on held-out tasks
6. Candidate beats baseline on expected verified utility
7. Statistical promotion gate passes
8. Safety metrics do not regress
9. Candidate is promoted as an immutable versioned policy
10. Restart loads the same promoted policy
11. Failed/incompatible policy always falls back safely
12. Results are reproducible from manifests and hashes
