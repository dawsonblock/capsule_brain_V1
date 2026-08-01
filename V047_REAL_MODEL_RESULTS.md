# v0.4.7 Real-Model Qualification Results — Qwen2.5-3B-Instruct

**Date:** 2026-07-31
**Run ID:** `v047_real_3b_001`
**Model:** Qwen/Qwen2.5-3B-Instruct (float16)
**Hardware:** RunPod RTX 5090 (32 GB VRAM)
**Evidence origin:** REAL_MODEL
**Protocol:** 0.4.7
**Commit:** `b44f35c`

---

## 1. Executive Summary

The v0.4.7 evaluation pipeline was executed end-to-end on real-model evidence
generated from Qwen2.5-3B-Instruct. The run produced a **scientifically honest
failure**: evidence is admissible and routing headroom exists, but the candidate
policy does not demonstrate causal effectiveness over the frozen baseline.

| Gate | Status | One-line summary |
|---|---|---|
| A0 — Evidence admissibility | **PASS** | 24/24 sub-gates passed; real-model evidence is valid |
| A1 — Routing headroom | **PASS** | Oracle exceeds baseline and sham by LCB > threshold |
| A2 — Candidate effectiveness | **FAIL** | Candidate is identical to baseline (delta = 0.0) |
| A3 — Robustness | **FAIL** | Single-seed run; blocked by design |
| A4 — Promotion | **FAIL** | Not promotion eligible |
| `legacy_gate_a_status` | **FAIL** | Correctly NOT PASS — the core v0.4.7 fix |

**Shadow eligible:** No
**Active eligible:** No

This result is exactly what the v0.4.7 specification demanded: a Gate A0
evidence-integrity pass is no longer misrepresented as Gate A success. The
pipeline honestly reports that the candidate has not been proven effective.

---

## 2. Why Gate A2 Failed

### 2.1 The Core Finding

The candidate policy produced by `run_local_gpu_scientific.py` is a simple
heuristic that computes average utility per action from experience rows and
assigns weights accordingly. On this benchmark, the heuristic selects the same
action as the frozen baseline for every single task:

| Metric | Value |
|---|---|
| `mean_delta` (candidate − baseline) | 0.000000 |
| `median_delta` | 0.000000 |
| `standard_error` | 0.000000 |
| `lower_bound` (95% CI) | 0.000000 |
| `upper_bound` (95% CI) | 0.000000 |
| `win_rate` | 0.00 (0 of 80 tasks) |
| `tie_rate` | 1.00 (80 of 80 tasks) |
| `loss_rate` | 0.00 |
| `practical_threshold` | 0.01 |
| `passes` | **False** |

The candidate ties baseline on all 80 tasks. The LCB (0.0) does not exceed the
practical-effect threshold (0.01), so Gate A2 correctly fails.

### 2.2 Candidate vs Sham (Passed)

The candidate does beat the sham control:

| Metric | Value |
|---|---|
| `mean_delta` (candidate − sham) | 3.130875 |
| `lower_bound` (95% CI) | 1.991126 |
| `practical_threshold` | 0.01 |
| `passes` | **True** |
| `win_rate` | 0.45 |
| `tie_rate` | 0.2875 |
| `loss_rate` | 0.2625 |

But Gate A2 requires beating **both** controls. The candidate beats sham but
not baseline, so the gate fails.

### 2.3 Root Cause

The heuristic candidate in `run_local_gpu_scientific.py` does not use the
`SupervisedRouterLearner` — it simply computes `action_utility[action] =
mean(utility)` and picks the highest-weight action. On this benchmark, that
produces the same routing decisions as the frozen `BaselinePolicyV2`, which
already routes competently. A real learned policy trained on counterfactual
experience with the `SupervisedRouterLearner` is needed to attempt Gate A2
PASS.

---

## 3. Why Gate A1 Passed

Gate A1 checks whether the benchmark offers sufficient routing headroom —
i.e., whether a perfect oracle could theoretically improve utility over
baseline and sham.

### Oracle vs Baseline

| Metric | Value |
|---|---|
| `mean_delta` | 0.600996 |
| `lower_bound` (95% CI) | 0.226620 |
| `practical_threshold` | 0.02 |
| `passes` | **True** |
| `win_rate` | 0.775 |
| `tie_rate` | 0.225 |
| `loss_rate` | 0.000 |

### Oracle vs Sham

| Metric | Value |
|---|---|
| `mean_delta` | 3.731871 |
| `lower_bound` (95% CI) | 2.692497 |
| `practical_threshold` | 0.02 |
| `passes` | **True** |
| `win_rate` | 0.725 |
| `tie_rate` | 0.275 |
| `loss_rate` | 0.000 |

The oracle wins 77.5% of tasks against baseline and 72.5% against sham. The
lower confidence bounds (0.227 and 2.69) both exceed the 0.02 threshold. This
means the benchmark does offer routing headroom — a sufficiently good learner
could theoretically improve utility. The failure is in the learner, not the
benchmark.

---

## 4. Why Gate A3 Failed

Gate A3 requires multi-seed replication: at least 3 generation seeds × 3
learner seeds = 9 candidate runs. The current evidence was generated with a
single generation seed, so Gate A3 is blocked by design.

| Check | Required | Actual | Status |
|---|---|---|---|
| Generation seeds | ≥ 3 | 1 | FAIL |
| Learner seeds | ≥ 3 | 1 | FAIL |
| Replication pass rate | ≥ 0.78 | 0.00 | FAIL |
| Action-distribution collapse | max_share < 0.98 | 1.00 | FAIL |
| Action coverage | ≥ 2 actions | 1 | FAIL |

The action-distribution collapse check also detected that the candidate selects
only one action (100% share), which exceeds the 0.98 maximum. This is
consistent with the Gate A2 finding — the candidate is identical to baseline,
which defaults to a single action.

---

## 5. Why Gate A0 Passed

Gate A0 (evidence admissibility) evaluates 24 sub-gates covering structural
completeness, evidence origin, provider authenticity, counterfactual
equivalence, split integrity, artifact lineage, and more.

All 24 sub-gates passed:

| Sub-Gate | Status | Observed |
|---|---|---|
| A0.1 Structural completeness | PASS | 0/0 match |
| A0.2 Evidence-origin authenticity | PASS | REAL_MODEL with model digest |
| A0.3 Scientific-claim eligibility | PASS | Real-model origin with model digest |
| A0.4 Cross-origin duplication | PASS | No duplicates across 2 packages |
| A0.5 Source provenance | PASS | source_hash present |
| A0.6 Historical identity | PASS | commit + source_hash |
| A0.7 Current analysis identity | PASS | Valid 64-char source-tree hash |
| A0.8 Cross-version lineage | PASS | Explicit lineage link |
| A0.9 Provider/model authenticity | PASS | provider_class=real_model |
| A0.10 Single generation identity | PASS | 1 revision, 1 generation |
| A0.11 Positive evidence weights | PASS | 30 weights, 0 negative |
| A0.12 Counterfactual equivalence | PASS | 60/60 equivalent |
| A0.13 Complete action matrix | PASS | 80/80 complete |
| A0.14 Verifier registry | PASS | 4/4 verifiers registered |
| A0.15 Utility consistency | PASS | 0 failures |
| A0.16 Split-access integrity | PASS | 0 violations |
| A0.17 Candidate serialization parity | PASS | parity=True |
| A0.18 Sham serialization parity | PASS | parity=True |
| A0.19 Task/split consistency | PASS | benchmark=80, split=80 |
| A0.20 Artifact lineage | PASS | 21/21 valid |
| A0.21 Metric consistency | PASS | 0 failures |
| A0.22 Safety evidence integrity | PASS | 0 failures |
| A0.23 No stale artifact reuse | PASS | 0 stale artifacts |
| A0.24 Oracle consistency | PASS | No discrepancy |

**Key point:** Gate A0 PASS means the evidence is admissible for evaluation.
It does NOT mean the learner is effective. This distinction is the core fix
of v0.4.7.

---

## 6. Safety Results

| Metric | Value |
|---|---|
| Safety status | **PASS** |
| n_safety_tasks | 10 |
| Severe violations (candidate) | 0 |
| Severe violations (baseline) | 0 |
| Total violations (candidate) | 0 |
| Total violations (baseline) | 0 |
| Severe violation increase | 0 |
| All critical cases pass | True |

Safety evidence is clean — no violations for any policy.

---

## 7. Family-Level Results

| Family | Status | Tasks | Groups | Cand-Baseline Delta | Cand-Sham Delta | CB LCB | CS LCB |
|---|---|---|---|---|---|---|---|
| unknown | PASS | 80 | 80 | 0.000000 | 3.130875 | 0.000000 | 2.180352 |

All 80 tasks fall into a single "unknown" family (the benchmark does not
assign task families in the current evidence format). The family has
sufficient support (80 tasks ≥ 30 threshold). The candidate-sham delta is
positive with LCB > 0, but the candidate-baseline delta is zero.

---

## 8. The Scientific Significance

### What this run proves

1. **The v0.4.7 gate hierarchy works correctly.** It separates evidence
   admissibility (A0) from causal effectiveness (A2) and honestly reports
   failure when the candidate doesn't beat baseline.

2. **The benchmark has routing headroom.** Gate A1 PASS means a perfect oracle
   could improve utility by ~0.6 over baseline. The opportunity exists — the
   learner just hasn't captured it.

3. **The candidate beats sham but not baseline.** This means the candidate is
   not merely random — it has some signal — but it doesn't improve on the
   already-competent frozen baseline.

4. **`legacy_gate_a_status` is "FAIL", not "PASS".** This is the core fix:
   in v0.4.6, a Gate A0 pass could be represented as `gate_a_status: "PASS"`,
   which was scientifically incorrect. v0.4.7 fixes this.

### What this run does NOT prove

1. It does not prove that AutoLearn is ineffective — the heuristic candidate
   is not the real `SupervisedRouterLearner`. A proper learned policy may
   capture the routing headroom that the oracle shows exists.

2. It does not prove that the benchmark is inadequate — Gate A1 PASS shows
   the benchmark has sufficient headroom.

3. It does not prove robustness — Gate A3 requires multi-seed replication
   which was not attempted.

### What is needed for Gate A2 PASS

1. Replace the heuristic candidate with a `SupervisedRouterLearner`-trained
   policy on the counterfactual experience data.
2. The learned policy must select different actions than baseline on at least
   some tasks where the oracle shows improvement is possible.
3. The LCB of the candidate-baseline delta must exceed 0.01.

### What is needed for Gate A3 PASS

1. Run evidence generation with 3 generation seeds (101, 202, 303).
2. Train candidate and sham under 3 learner seeds (11, 22, 33).
3. At least 7 of 9 replicates must have nonnegative candidate-vs-sham deltas.
4. No catastrophic sign reversals (delta < -0.02).

---

## 9. Reproduction

```bash
# On a GPU machine (RTX 5090 or similar):
git clone https://github.com/dawsonblock/capsule_brain_V1.git
cd capsule_brain_V1
pip install torch transformers accelerate tqdm pyyaml

# Generate evidence (uses existing run_local_gpu_scientific.py):
PYTHONPATH=src python -m qualification.autolearn_v04.run_local_gpu_scientific \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir scientific_evidence_3b \
  --batch-size 8

# Run v0.4.7 evaluation (all 5 gates):
PYTHONPATH=src python -c "
from qualification.autolearn_v04.v047.orchestrator import run_v047_evaluation
result = run_v047_evaluation(
    evidence_dir='scientific_evidence_3b',
    output_dir='v047_output',
    run_id='v047_real_3b_001',
    repo_root='.',
    force=True,
)
"

# View the machine-readable verdict:
cat v047_output/v047_real_3b_001/MACHINE_VERDICT.json

# View the full report:
cat v047_output/v047_real_3b_001/FINAL_REPORT.md
```

---

## 10. Non-Negotiable Scientific Rule

> Valid evidence is not positive evidence.
> Oracle headroom is not learned improvement.
> Candidate improvement over baseline is not causal learning
> unless the candidate also beats a properly matched sham control.
> A positive point estimate is not a pass
> unless the lower confidence bound exceeds the prespecified
> practical-effect threshold.
> A single run is not robust evidence.
> A synthetic fixture is not a real-model result.
> A passive hidden-state probe is not latent reasoning.

This run obeys every one of these rules. The result is an honest failure —
the most valuable kind.
