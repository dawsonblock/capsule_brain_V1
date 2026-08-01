# v0.4.7 Real-Model Qualification Results — Qwen2.5-3B-Instruct (Corrected)

**Date:** 2026-08-01
**Run ID:** `v047_real_3b_v3_001`
**Model:** Qwen/Qwen2.5-3B-Instruct (float16)
**Hardware:** RunPod RTX 4090 (24 GB VRAM)
**Evidence origin:** REAL_MODEL
**Protocol:** 0.4.7
**Commit:** `22284bb`

---

## 1. Executive Summary

This is the corrected run that fixes the six experiment-integrity defects
identified in the build 17 analysis. The previous run (`v047_real_3b_v2_002`)
claimed "ALL GATES PASS" but that claim was invalid due to split contamination,
fake generation seeds, incorrect sham delta calculation, and other issues.

This corrected run produces a **scientifically honest FAIL**:

| Gate | Status | One-line summary |
|---|---|---|
| A0 — Evidence admissibility | **FAIL** | v0.4.6 Gate A failed (candidate=sham) |
| A1 — Routing headroom | **FAIL** | Inherits from A0 |
| A2 — Candidate effectiveness | **FAIL** | Candidate beats baseline but NOT sham (delta=0.0) |
| A3 — Robustness | **FAIL** | 0/3 replicates pass (candidate_vs_sham never passes) |
| A4 — Promotion | **FAIL** | shadow_eligible=false, active_eligible=false |
| `legacy_gate_a_status` | **FAIL** | Correctly FAIL |

**Shadow eligible:** No
**Active eligible:** No

---

## 2. What Was Fixed from Previous Run

### 2.1 Split contamination fixed (P0-1)

Policy results (baseline, candidate, sham, oracle) now evaluate ONLY on
D_test (20 tasks). Experience, validation, OOD, and safety rows are excluded.
Hard assertions prevent test/experience and test/validation overlap.

### 2.2 Metadata preserved in result rows (P0-2)

Every result row now includes `split`, `family`, and `task_group_id`.
Family evaluation can now properly distinguish task families (4 families
detected: direct_answer, memory_required, tool_required, workflow_required).

### 2.3 Candidate-vs-sham delta fixed (P0-3)

Replicate records now compute `candidate_vs_sham_delta` independently from
`candidate_vs_baseline_delta`. The previous code incorrectly copied the
baseline delta to the sham delta.

### 2.4 Proper stratified label permutation sham (P0-4)

Sham now permutes labels across ALL examples stratified by family,
preserving marginal label statistics while destroying state-target
association. The previous per-task shuffle was not a true permutation.

### 2.5 Fake generation seeds removed (P0-5)

Generation is deterministic (greedy decoding). We now honestly report
`generation_deterministic=true` with 1 generation seed. Gate A3 handles
this correctly (requires >= 1 gen seed when deterministic).

### 2.6 Promotion semantics fixed (P0-6)

CLI now routes through `evaluate_gate_a4()` module. `active_eligible` is
ALWAYS False after offline qualification, matching the intended
shadow/active promotion architecture.

### 2.7 Source provenance fixed (P0-7)

`source_tree_sha256` now hashes actual Python source files (src/,
qualification/), not the benchmark manifest JSON.

### 2.8 Dependency identity fixed (P0-8)

`dependency_identity` now comes from runtime inspection
(`torch.__version__`, `transformers.__version__`, GPU name, CUDA version)
instead of hardcoded strings.

### 2.9 Safety evaluation fixed (P0-10)

CLI now loads `candidate_safety_results.json`, `baseline_safety_results.json`,
and `sham_safety_results.json` for proper safety comparison across all
three policies.

---

## 3. Detailed Gate Results

### Gate A0 — Evidence Admissibility

- **Status:** FAIL
- **Reason:** v0.4.6 Gate A failed (candidate_vs_sham_lcb: fail, delta=0.0)
- **Sub-gates:** 24 checks, evidence_origin=REAL_MODEL

### Gate A1 — Routing Headroom (D_test only, 20 tasks)

- **Status:** FAIL (inherits from A0)
- **Oracle vs Baseline:**
  - Mean delta: 2.288
  - 95% LCB: 0.781
  - Win rate: 100%
  - Threshold: 0.02
  - **PASS** — oracle clearly beats baseline on D_test

### Gate A2 — Candidate Causal Effectiveness (D_test only, 20 tasks)

- **Status:** FAIL
- **Candidate vs Baseline:**
  - Mean delta: 2.288
  - 95% LCB: 0.781
  - Win rate: 100%
  - Threshold: 0.01
  - **PASS** — candidate clearly beats baseline on D_test
- **Candidate vs Sham:**
  - Mean delta: 0.000
  - 95% LCB: 0.000
  - Win rate: 0%
  - **FAIL** — candidate does NOT beat sham

### Gate A3 — Robustness and Replication

- **Status:** FAIL
- **Replicate summary:**
  - n_replicates: 3 (1 gen seed x 3 learner seeds)
  - generation_deterministic: true
  - Replication pass rate: 0% (0/3)
  - Catastrophic reversals: 0
  - Seed diversity: OK
- **Family summary:**
  - n_sufficient: 0
  - n_insufficient: 4
  - Critical regressions: 0
- **Collapse check:** PASS
- **Safety check:** PASS

### Gate A4 — Promotion Eligibility

- **Status:** FAIL
- **Shadow eligible:** No
- **Active eligible:** No
- **Blocking reasons:** Gate A0-A3 did not pass

---

## 4. The Key Scientific Finding

**Gate A failed. The observed candidate-over-baseline improvement cannot
be attributed to learned causal state→action policy improvement because
the sham policy reproduces the gain. Current evidence indicates the
benchmark permits family-level shortcut learning.**

The sham with stratified label permutation achieves identical performance
to the candidate because:

1. The features encode family identity directly
   (`workflow_capability_match = family == "workflow_required"`,
   `hit_count = 1 if family == "memory_required"`)

2. Within each family, the best action is constant
   (all tool_required → CALL_TOOL, all memory_required → RETRIEVE_MEMORY, etc.)

3. Stratified permutation within family preserves the family→action mapping

4. Both candidate and sham learn the same family→action mapping

5. The observed advantage is reproducible by the sham and is consistent
   with exploitation of family-level structure

This is exactly what a good negative control is supposed to expose. The
experiment has demonstrated that the policy learner can exploit task-family
structure, but it has not demonstrated causal state-conditioned policy
learning.

---

## 5. What This Run Proves

1. The corrected experiment infrastructure works correctly:
   - D_test-only evaluation
   - Proper stratified label permutation sham
   - Honest generation_deterministic reporting
   - Correct candidate-vs-sham delta calculation
   - Proper promotion semantics (active=false)

2. The candidate (SupervisedRouterLearner) beats the baseline
   (BaselinePolicyV3) on D_test: delta=2.288, LCB=0.781, win_rate=100%

3. The candidate does NOT beat the sham: delta=0.000

4. The advantage over baseline comes from family→action mapping, not
   causal state→action learning

5. The gate framework correctly detects this and returns FAIL

---

## 6. What Is Needed Next (P1 — Scientific Strength)

1. **Feature ablation**: Remove family-proxy features and rerun. If
   performance collapses, the current experiment is measuring engineered
   label leakage.

2. **Stronger sham**: The current stratified permutation preserves
   family→action mapping. Need a sham that destroys this mapping while
   preserving marginal label statistics across the full dataset.

3. **Prompt-only features**: Use only features derived naturally from
   prompt text, not from setup_spec or family labels.

4. **Stronger statistical baseline**: Compare against logistic regression
   trained directly on task labels, not just keyword heuristics.

5. **Larger test set**: 20 test tasks is too small for convincing
   generalization evidence.

6. **More diverse families**: Ensure multiple tasks per family have
   different best actions, so family identity alone doesn't determine
   the optimal route.

---

## 7. Model Performance

- **Model:** Qwen/Qwen2.5-3B-Instruct (float16)
- **VRAM:** 6.42 GB
- **Total tasks:** 80 (30 experience, 10 validation, 20 test, 10 ood, 10 safety)
- **Total counterfactuals:** 320 (4 actions x 80 tasks)
- **Verified success:** 49/320 (15.3%)
- **Mean latency:** 1073ms
- **Mean tokens:** 269
- **Total elapsed:** 368s

---

## 8. D_test Performance (20 tasks)

| Policy | Mean utility | n_rows |
|---|---|---|
| Baseline (BaselinePolicyV3) | -3.005 | 20 |
| Candidate (SupervisedRouterLearner) | -0.717 | 20 |
| Sham (permuted-label) | -0.717 | 20 |
| Oracle (best action) | -0.717 | 20 |

Candidate = Sham = Oracle (all learn the same family→action mapping).
Baseline is worse because its keyword heuristics are weaker than
the learned family classifier.

---

## 9. Reproducibility

- **Commit:** `22284bb`
- **Model:** Qwen/Qwen2.5-3B-Instruct (HuggingFace, float16)
- **Generation:** Greedy decoding (do_sample=False, deterministic)
- **Task seed:** 42
- **Learner seeds:** [11, 22, 33]
- **Generation seed:** 101 (single deterministic run)
- **Hardware:** RunPod RTX 4090, 24GB VRAM
- **Dependencies:** Captured from runtime inspection in source_provenance.json
