# v0.4.7 Real-Model Qualification Results — Qwen2.5-3B-Instruct

**Date:** 2026-08-01
**Protocol:** 0.4.7
**Model:** Qwen/Qwen2.5-3B-Instruct (float16)
**Hardware:** RunPod RTX 4090 (24 GB VRAM)
**Evidence origin:** REAL_MODEL

---

## Build 18 (v3) — Corrected Run with Integrity Fixes

**Run ID:** `v047_real_3b_v3_001`
**Commit:** `22284bb`

### Result: ALL GATES FAIL

| Gate | Status |
|---|---|
| A0 | FAIL |
| A1 | FAIL |
| A2 | FAIL |
| A3 | FAIL |
| A4 | FAIL |

### Key metrics (D_test, 20 tasks)

| Policy | Mean utility |
|---|---|
| Baseline | -3.005 |
| Candidate | -0.717 |
| Sham | -0.717 |
| Oracle | -0.717 |

Candidate vs baseline: delta=2.288, LCB=0.781 — **PASS**
Candidate vs sham: delta=0.000 — **FAIL**

### Conclusion

Gate A failed. The observed candidate-over-baseline improvement cannot
be attributed to learned causal state→action policy improvement because
the sham policy reproduces the gain. Current evidence indicates the
benchmark permits family-level shortcut learning.

---

## Build 19 (v5) — Shortcut-Proof Benchmark with Within-Family Variation

**Run ID:** `v047_real_3b_v5_001`
**Commit:** `20ab70e`

### Changes from Build 18

1. **Within-family action crossovers**: 8 new task builders create tasks
   with the surface form of one family but requiring a DIFFERENT optimal
   action. Every family now has ∃ x_i, x_j ∈ F_k: a*(x_i) ≠ a*(x_j).

2. **Prompt-only features**: All family-proxy features removed from
   `_make_state`. Features derived only from prompt text keywords.

3. **Feature-family diagnostic**: Trains g(h(x))→family to audit leakage.

4. **Stronger sham suite**: Added random-matched, feature-permutation,
   and family-only shams alongside the stratified label permutation.

5. **Leave-family-out evaluation**: Train on k-1 families, test on held-out.

### Result: ALL GATES FAIL

| Gate | Status |
|---|---|
| A0 | BLOCKED |
| A1 | FAIL |
| A2 | FAIL |
| A3 | FAIL |
| A4 | FAIL |

### Key metrics (D_test, 23 tasks)

| Policy | Mean utility |
|---|---|
| Baseline | -3.033 |
| Candidate | -1.425 |
| Sham | -1.428 |
| Oracle | -1.425 |

Candidate vs baseline: delta=0.327 — candidate still beats baseline
Candidate vs sham: delta=0.003 — candidate does NOT meaningfully beat sham

### Feature-family diagnostic

- Family predictability accuracy: **0.486** (down from 0.800 in Build 18)
- Verdict: **OK** (below 0.8 threshold)
- The prompt-only features no longer strongly encode family identity

### Sham suite results

| Sham type | Train accuracy |
|---|---|
| Stratified label permutation | 0.935 (down from 1.000) |
| Random matched | 0.677 |
| Feature permutation | 0.677 |

The within-family crossovers made the sham's job harder (0.935 vs 1.000),
but it still nearly matches the candidate on D_test.

### Leave-family-out results

| Held-out family | n_train | Test utility |
|---|---|---|
| direct_answer | 18 | -4.080 |
| memory_required | 26 | -2.689 |
| tool_required | 26 | -4.210 |
| workflow_required | 23 | -4.424 |

All held-out families show poor transfer vs candidate on full test (-1.425).
The policy still relies heavily on family-level structure.

### Conclusion

Gate A failed. The within-family crossovers successfully:
- Reduced feature-family predictability (0.800 → 0.486)
- Prevented the sham from perfectly memorizing training data (1.000 → 0.935)
- Opened a tiny candidate-vs-sham gap (0.000 → 0.003)

But the candidate still does not meaningfully beat the sham. The current
learner (SupervisedRouterLearner with nearest-centroid classification on
handcrafted features) is not powerful enough to exploit the within-family
action variation. The sham can still approximate the candidate's
performance because the prompt-derived features don't provide enough
signal to distinguish which action is optimal WITHIN a family.

### What this proves

1. The benchmark redesign works: family-level shortcuts are harder
2. The feature-family diagnostic confirms reduced leakage
3. The sham suite provides meaningful negative controls
4. The leave-family-out evaluation confirms family dependence
5. The current learner architecture is insufficient for causal
   state-conditioned routing — it needs features that can distinguish
   within-family action optimality, or a more expressive model

### What is needed next

The benchmark is now structurally harder, but the features and learner
are too weak to solve it. Options:

1. **Richer features**: Add features that can distinguish within-family
   action optimality (e.g., prompt embeddings, semantic features)
2. **More expressive learner**: Move beyond nearest-centroid to a
   learner that can capture within-family decision boundaries
3. **More within-family examples**: Increase n_experience so the learner
   sees more examples of within-family action variation
4. **Matched-pair flip accuracy**: Add the direct test of whether the
   policy responds to the causal feature rather than the task class

The scientifically defensible conclusion remains: Gate A has not passed.
The experiment infrastructure is now capable of falsifying the hypothesis,
which is progress, but the current learner cannot solve the shortcut-proof
benchmark.
