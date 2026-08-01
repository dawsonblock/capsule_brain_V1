# Plan to Pass Gate A2 and Gate A3 — v0.4.7 Real-Model Qualification

**Status:** PLAN ONLY — not implemented
**Date:** 2026-07-31
**Author:** Devin

---

## Executive Summary

**It is possible to pass Gate A2 and Gate A3.** The root cause of the current
failure is that `run_local_gpu_scientific.py` uses a **heuristic global-average
candidate** that picks one action for all tasks — identical to the hardcoded
baseline (always `ANSWER_DIRECT`). The real `SupervisedRouterLearner` exists in
the codebase but is not wired into the scientific pipeline.

The benchmark has 4 task families (direct_answer, memory_required,
tool_required, workflow_required) with balanced distribution (~15 tasks each).
The oracle achieves 100% success by routing each task to its family-appropriate
action. Baseline achieves only 25% because it always picks `ANSWER_DIRECT`.
There is 5.25 utility points of headroom. The 17 features in `FeatureExtractor`
capture the necessary signals (keyword counts, memory similarity, tool/workflow
availability).

The fix is to replace the heuristic candidate with the real
`SupervisedRouterLearner`, replace the hardcoded baseline with
`BaselinePolicyV3`, replace the random sham with proper permuted-label
training, and add multi-seed replication for Gate A3.

---

## Root Cause Analysis

### Why Gate A2 Failed

The candidate policy in `run_local_gpu_scientific.py` (lines 460-476) computes
a **global average utility per action** across all experience rows, then picks
the single action with the highest average for every task:

```python
# Current (broken) candidate — lines 460-476
action_utility: dict[str, list[float]] = {}
for row in experience_rows:
    action_utility.setdefault(row["action_id"], []).append(row["utility"])
candidate_policy_weights = {
    action: round(sum(utils) / len(utils), 4) if utils else 0.0
    for action, utils in action_utility.items()
}
```

Since `ANSWER_DIRECT` has the highest global average utility (direct_answer
tasks succeed with it), the candidate picks `ANSWER_DIRECT` for every task.
The baseline (line 499) is also hardcoded to `ANSWER_DIRECT`. Result:
`mean_delta = 0.0`, `tie_rate = 1.0`, Gate A2 FAIL.

### Why Gate A3 Failed

Only one generation seed was used. Gate A3 requires ≥ 3 generation seeds ×
≥ 3 learner seeds = ≥ 9 replicates. Single-seed runs are BLOCKED by design.

---

## Plan: Gate A2 Fix (6 steps)

### Step 1: Build training dataset from counterfactual outcomes

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`
**Location:** After counterfactual execution (~line 440), before candidate training

Add a call to `build_dataset` to create the train/validation splits from the
counterfactual outcomes. This produces `dataset_train.json` and
`dataset_validation.json` with feature vectors and utility-weighted labels.

```python
from qualification.autolearn_v04.build_dataset import build_dataset
# Build dataset from counterfactual outcomes
build_dataset(config)  # Creates dataset_train.json, dataset_validation.json
```

**Risk:** The `build_dataset` function may expect a `QualificationConfig` object
that the simplified `run_local_gpu_scientific.py` doesn't construct. May need
to construct a minimal config or refactor `build_dataset` to accept raw paths.

### Step 2: Train candidate with SupervisedRouterLearner

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`
**Location:** Replace lines 460-476

Replace the heuristic global-average with a call to `train_candidate`, which
uses `SupervisedRouterLearner` (multinomial logistic regression on 17 features,
400 epochs, weighted cross-entropy, L2 regularization):

```python
from qualification.autolearn_v04.train_candidate import train_candidate
candidate_policy, candidate_training, candidate_prov = train_candidate(config)
```

The `SupervisedRouterLearner` will learn to route:
- Tasks with high `tool_required_keywords` → `CALL_TOOL`
- Tasks with high `retrieval_indicators` + `semantic_memory_similarity` → `RETRIEVE_MEMORY`
- Tasks with high `code_indicators` + `workflow_available` → `START_WORKFLOW`
- Otherwise → `ANSWER_DIRECT`

This produces a **task-specific** policy that selects different actions for
different tasks, unlike the current global-average heuristic.

**Risk:** `train_candidate` may depend on artifacts (split_manifest.json,
dataset files) that the simplified pipeline doesn't create. May need to add
intermediate steps or refactor.

### Step 3: Replace hardcoded baseline with BaselinePolicyV3

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`
**Location:** Replace lines 495-511

The current baseline is hardcoded to `ANSWER_DIRECT`:
```python
# Current (broken) baseline — line 499
task_outcomes = [o for o in outcomes if o["task_id"] == tid and o["action_id"] == "ANSWER_DIRECT"]
```

Replace with `BaselinePolicyV3`, which uses keyword heuristics with fixed
thresholds (tool_signal ≥ 1.5, workflow_signal ≥ 1.5, memory_signal ≥ 1.5):

```python
from capsule_brain.autolearn.baseline import BaselinePolicyV3
from qualification.autolearn_v04.evaluate import evaluate_policy_on_split

baseline_policy = BaselinePolicyV3()
baseline_results = evaluate_policy_on_split(
    config, split="test", policy=baseline_policy, label="baseline"
)
```

**Important:** On the current scientific benchmark, `BaselinePolicyV3` also
defaults to `ANSWER_DIRECT` for most tasks because the keyword thresholds (1.5)
are too high for the benchmark's prompt style. This means baseline will still
get ~25% success rate, and the learned candidate should beat it by routing
correctly to memory/tool/workflow actions.

**Risk:** If `BaselinePolicyV3` happens to route correctly on some tasks, the
candidate's advantage shrinks. But oracle shows 77.5% win rate over baseline,
so there's plenty of headroom.

### Step 4: Replace random sham with proper permuted-label training

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`
**Location:** Replace lines 478-493

The current sham uses random weights:
```python
# Current (broken) sham — lines 478-493
sham_policy_weights = {action: random.random() for action in ACTIONS}
```

Replace with `train_sham`, which trains `SupervisedRouterLearner` on
**permuted labels** (same features, shuffled action labels). This ensures the
sham has the same model capacity and feature access as the candidate, but
learns a random mapping:

```python
from qualification.autolearn_v04.train_sham import train_sham
sham_policy, sham_training, sham_prov = train_sham(config, seed=42)
```

**Risk:** `train_sham` may have the same dependency issues as `train_candidate`.

### Step 5: Use evaluate_policy_on_split for all evaluations

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`
**Location:** Replace lines 495-576

Replace the manual evaluation loops with `evaluate_policy_on_split` for
consistent evaluation across candidate, baseline, sham, and oracle:

```python
from qualification.autolearn_v04.evaluate import evaluate_policy_on_split

candidate_results = evaluate_policy_on_split(config, split="test", policy=candidate_policy, label="candidate")
baseline_results = evaluate_policy_on_split(config, split="test", policy=baseline_policy, label="baseline")
sham_results = evaluate_policy_on_split(config, split="test", policy=sham_policy, label="sham")
oracle_results = evaluate_policy_on_split(config, split="test", policy=None, label="oracle", oracle=True)
```

**Risk:** `evaluate_policy_on_split` may expect the model provider to be
available for re-evaluation. Since we're using pre-computed counterfactual
outcomes, we need to ensure the evaluation uses the stored outcomes rather than
re-running the model.

### Step 6: Verify Gate A2 pass

After re-running the pipeline:
- `candidate_vs_baseline` should show `mean_delta > 0` with `LCB > 0.01`
- `candidate_vs_sham` should show `mean_delta > 0` with `LCB > 0.01`
- Gate A2 status should be `PASS`

**Expected outcome:** Candidate routes ~75% of tasks correctly (all non-direct
tasks), baseline routes ~25% correctly (only direct tasks). Candidate utility
should be ~7-8 vs baseline ~3.75. Delta ~3-4, LCB well above 0.01.

---

## Plan: Gate A3 Fix (3 steps)

### Step 7: Implement multi-seed evidence generation

**File:** New function in `qualification/autolearn_v04/run_local_gpu_scientific.py`

Add a `--multi-seed` flag that runs evidence generation with 3 generation
seeds (e.g., 101, 202, 303). Each seed produces a separate counterfactual
outcome set:

```python
GENERATION_SEEDS = [101, 202, 303]
LEARNER_SEEDS = [11, 22, 33]

for gen_seed in GENERATION_SEEDS:
    # Re-run counterfactual execution with this seed
    # Produces counterfactual_outcomes_seed{gen_seed}.jsonl
```

**Risk:** This triples the GPU time for evidence generation (~30 min → ~90 min
on RTX 5090). The model generates 4 counterfactual outcomes per task × 80
tasks × 3 seeds = 960 generations.

### Step 8: Train candidate and sham under multiple learner seeds

For each generation seed, train candidate and sham under 3 learner seeds:

```python
for gen_seed in GENERATION_SEEDS:
    for learner_seed in LEARNER_SEEDS:
        candidate, _ = train_candidate(config, seed=learner_seed)
        sham, _ = train_sham(config, seed=learner_seed)
        # Evaluate on test split
        # Store results as replicate_{gen_seed}_{learner_seed}
```

This produces 9 candidate-sham pairs. Gate A3 requires ≥ 7 of 9 to have
nonnegative candidate-vs-sham delta.

**Risk:** Training is fast (CPU-only logistic regression), but evaluation
requires the pre-computed outcomes for each generation seed.

### Step 9: Verify Gate A3 pass

After multi-seed runs:
- `n_generation_seeds` should be 3 (≥ 3 required)
- `n_learner_seeds` should be 3 (≥ 3 required)
- `replication_pass_rate` should be ≥ 0.78 (≥ 7 of 9)
- `catastrophic_reversals` should be 0
- Action-distribution collapse should pass (candidate uses multiple actions)

**Expected outcome:** Since the candidate consistently beats sham (LCB=1.99
in single-seed), all 9 replicates should have positive candidate-vs-sham delta.
Gate A3 should PASS.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `train_candidate`/`train_sham` have unmet dependencies in the simplified pipeline | Medium | Refactor to accept raw paths instead of `QualificationConfig`, or construct a minimal config |
| `evaluate_policy_on_split` tries to re-run the model instead of using stored outcomes | Medium | Add a `use_stored_outcomes=True` flag or ensure the evaluation reads from counterfactual_outcomes.jsonl |
| `BaselinePolicyV3` routes correctly on some non-direct tasks, shrinking candidate advantage | Low | Oracle shows 77.5% win rate; even if baseline gets some right, candidate should still beat it |
| Multi-seed evidence generation takes too long | Low | RTX 5090 is fast; 960 generations at ~3 sec each = ~48 min |
| `train_sham` with permuted labels accidentally learns a useful pattern | Very Low | Permuted labels destroy the feature-action mapping by design |
| Action-distribution collapse check still fails | Low | With `SupervisedRouterLearner`, the candidate will use all 4 actions, not just one |

---

## What Will NOT Work

1. **Tweaking the heuristic candidate's weights** — any global-average policy
   will pick one action for all tasks. The fundamental problem is that it's
   task-agnostic.

2. **Lowering the Gate A2 threshold** — the threshold (0.01) is already very
   low. The problem is `mean_delta = 0.0`, not a marginal miss.

3. **Using a different baseline** — the baseline must be the frozen
   `BaselinePolicyV3` for scientific validity. Changing it to make the
   candidate look better would be cheating.

4. **Running more bootstrap resamples** — the confidence interval is already
   tight (SE=0.0). More resamples won't change the result.

---

## Estimated Effort

| Step | Effort | GPU time |
|---|---|---|
| Steps 1-6 (Gate A2) | ~2-3 hours coding | ~30 min GPU (re-run evidence) |
| Steps 7-9 (Gate A3) | ~1-2 hours coding | ~90 min GPU (multi-seed) |
| Total | ~3-5 hours | ~2 hours GPU |

---

## Scientific Validity Checklist

- [ ] Candidate uses `SupervisedRouterLearner` (not heuristic)
- [ ] Baseline uses `BaselinePolicyV3` (not hardcoded ANSWER_DIRECT)
- [ ] Sham uses permuted-label training (not random weights)
- [ ] Evaluation uses `evaluate_policy_on_split` (not manual loops)
- [ ] Multi-seed: 3 generation seeds × 3 learner seeds
- [ ] No gate thresholds are lowered
- [ ] No evidence is fabricated or cherry-picked
- [ ] All artifacts are checksummed and lineage-tracked

---

## Conclusion

**Passing Gate A2 and Gate A3 is possible.** The infrastructure exists
(`SupervisedRouterLearner`, `train_candidate`, `train_sham`,
`evaluate_policy_on_split`, `BaselinePolicyV3`). The benchmark has sufficient
headroom (oracle 9.0 vs baseline 3.75). The features capture the necessary
signals. The only reason it failed is that `run_local_gpu_scientific.py` used
a simplified heuristic instead of the real learner.

The fix is to wire the real learner into the scientific pipeline and add
multi-seed replication. No scientific shortcuts are needed — just use the
tools that already exist.
