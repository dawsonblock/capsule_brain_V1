# Implementation Summary — Capsule Brain 2.15.11

**Release:** 2.15.11
**AutoLearn:** 0.3.10
**AutoLearn Qualification:** 0.4.7
**Qualification Protocol:** 0.4.7
**Date:** 2026-07-31
**Commit:** `b092e7a`

---

## Overview

This release implements the **evidence-complete causal gate hierarchy** that
separates evidence admissibility from causal effectiveness. The core scientific
defect in v0.4.6 was that a Gate A0 evidence-integrity pass could be represented
as `gate_a_status: "PASS"`, conflating "evidence is valid enough to evaluate"
with "the learner has been proven effective."

v0.4.7 corrects this by replacing the ambiguous Gate A structure with five
explicit staged gates: A0 (admissibility), A1 (routing headroom), A2 (candidate
causal effectiveness), A3 (robustness and replication), and A4 (promotion
eligibility).

**No new cognitive architecture was added.** The release is scoped to fixing
evidence and qualification defects before any new cognitive architecture.

---

## Files Added

### v0.4.7 Qualification Module (18 new files)

| File | Purpose |
|---|---|
| `qualification/autolearn_v04/v047/__init__.py` | Package init with protocol version |
| `qualification/autolearn_v04/v047/gate_schema.py` | Structured `QualificationVerdict` with conservative `legacy_gate_a_status` |
| `qualification/autolearn_v04/v047/config.py` | Versioned `QualificationConfigV047` with all gate thresholds |
| `qualification/autolearn_v04/v047/statistics.py` | Paired cluster bootstrap by task group |
| `qualification/autolearn_v04/v047/gate_a0.py` | Wrapper for v0.4.6 Gate A0 (24 sub-gates) |
| `qualification/autolearn_v04/v047/gate_a1.py` | Routing headroom evaluation (oracle vs baseline/sham) |
| `qualification/autolearn_v04/v047/gate_a2.py` | Candidate causal effectiveness (must beat BOTH controls) |
| `qualification/autolearn_v04/v047/gate_a3.py` | Robustness and replication (multi-seed, family-stratified) |
| `qualification/autolearn_v04/v047/gate_a4.py` | Promotion eligibility (shadow/active) |
| `qualification/autolearn_v04/v047/family_evaluation.py` | Per-family statistics with support thresholds |
| `qualification/autolearn_v04/v047/collapse_checks.py` | Action-distribution collapse detection |
| `qualification/autolearn_v04/v047/safety_evaluation.py` | Safety evidence with severe violation checks |
| `qualification/autolearn_v04/v047/evidence_enforcement.py` | REAL_MODEL/SYNTHETIC/UNAVAILABLE enforcement |
| `qualification/autolearn_v04/v047/run_directory.py` | Strict immutable run-directory contract |
| `qualification/autolearn_v04/v047/artifact_dag.py` | Typed artifact dependency graph with cycle detection |
| `qualification/autolearn_v04/v047/promotion_manifest.py` | Cryptographic promotion binding |
| `qualification/autolearn_v04/v047/report.py` | 19-section report with precise language rules |
| `qualification/autolearn_v04/v047/cli.py` | 6-command CLI with correct exit codes |
| `qualification/autolearn_v04/v047/orchestrator.py` | Full 5-gate evaluation pipeline |

### Configuration (1 new file)

| File | Purpose |
|---|---|
| `qualification/configs/qwen25_3b_gate_a.yaml` | v0.4.7 YAML config for Qwen2.5-3B Gate A |

### Tests (2 new files)

| File | Tests | Purpose |
|---|---|---|
| `tests/unit/test_v047_gate_separation.py` | 31 | Gate separation, statistical, evidence enforcement, safety |
| `tests/unit/test_v047_artifact_integrity.py` | 23 | Artifact integrity, promotion binding, serialization, reporting |

---

## Files Changed

| File | Change |
|---|---|
| `src/capsule_brain/version.py` | Version bump + component version constants |
| `pyproject.toml` | Version bump to 2.15.11 |
| `qualification/QUALIFICATION_MANIFEST.json` | Version bump + active_analysis → v047 |
| `qualification/autolearn_v04/v045/analysis_identity.py` | Expected version constants updated |
| `qualification/autolearn_v04/v043/config.py` | Version constants updated |
| `qualification/autolearn_v04/run_local_gpu_scientific.py` | Version strings in evidence manifests updated |
| `tests/unit/test_v046_evidence_repair.py` | Version identity assertions updated |
| `tests/unit/test_v045_evidence_repair.py` | Version identity assertions updated |
| `tests/unit/test_v044_repair.py` | Version identity assertions updated |
| `tests/unit/test_v04_qualification.py` | Version identity assertions updated |

---

## Files Removed or Archived

No files were removed or archived. The v0.4.6 modules remain in place and are
reused by v0.4.7 (Gate A0 wraps the existing 24 sub-gate evaluation).

---

## Behavioral Changes

### 1. Gate A0 PASS no longer implies Gate A PASS

The `legacy_gate_a_status` field is computed conservatively:
```
legacy_gate_a_status = "PASS" if (A2 == PASS and A3 == PASS) else "FAIL_OR_BLOCKED"
```
A Gate A0 pass must NEVER map to legacy Gate A success.

### 2. Candidate must beat BOTH baseline and sham

Gate A2 passes only when:
- `LCB_95%(ΔU_candidate_vs_baseline) > ε_0` (default 0.01)
- `LCB_95%(ΔU_candidate_vs_sham) > ε_S` (default 0.01)

A positive point estimate does NOT pass when the lower confidence bound does not
exceed the threshold.

### 3. Gate A1 checks routing headroom

Gate A1 compares oracle against baseline and sham. A Gate A1 failure means the
benchmark offers insufficient routing headroom — it does NOT mean the learner is
defective.

### 4. Gate A3 requires replication

Gate A3 requires at least 3 generation seeds × 3 learner seeds = 9 candidate
runs. Single-seed runs are BLOCKED. At least 7 of 9 replicates should have
nonnegative candidate-vs-sham point estimates.

### 5. Gate A4 separates shadow and active eligibility

- `shadow_eligible`: all gates pass + REAL_MODEL evidence
- `active_eligible`: requires post-shadow validation (not yet implemented)

### 6. Synthetic evidence cannot satisfy real-model promotion

`require_scientific_evidence()` raises `ValueError` when synthetic evidence is
used for a real-model claim. `can_promote()` returns `False` for synthetic
evidence when `require_real_model=True`.

### 7. Safety failures block promotion

Severe safety violation increases block promotion. Critical case failures block
promotion. Safety failures are NOT averaged away through general utility
improvements.

### 8. CLI exit codes are scientifically honest

| Code | Meaning |
|---|---|
| 0 | Requested operation passed |
| 1 | Scientific gate failed |
| 2 | Blocked or incomplete evidence |
| 3 | Invalid configuration |
| 4 | Artifact-integrity failure |
| 5 | Runtime execution failure |

Exit code 0 is NOT returned for blocked qualification.

---

## Schema Changes

### Structured Verdict (replaces `gate_a_status: "PASS"`)

```json
{
  "protocol_version": "0.4.7",
  "run_id": "string",
  "evidence_origin": "REAL_MODEL",
  "gate_a0_admissibility": {"status": "PASS", "reasons": [], "checks": {}},
  "gate_a1_headroom": {"status": "PASS", "oracle_vs_baseline": {}, "oracle_vs_sham": {}},
  "gate_a2_effectiveness": {"status": "FAIL", "candidate_vs_baseline": {}, "candidate_vs_sham": {}},
  "gate_a3_robustness": {"status": "BLOCKED", "replicate_summary": {}, "family_summary": {}},
  "gate_a4_promotion": {"status": "BLOCKED", "shadow_eligible": false, "active_eligible": false}
}
```

### Machine-Readable Verdict

```json
{
  "release": "2.15.11",
  "autolearn": "0.3.10",
  "qualification": "0.4.7",
  "protocol": "0.4.7",
  "gate_a0_admissibility": "PASS|FAIL|BLOCKED",
  "gate_a1_headroom": "PASS|FAIL|BLOCKED",
  "gate_a2_effectiveness": "PASS|FAIL|BLOCKED",
  "gate_a3_robustness": "PASS|FAIL|BLOCKED",
  "gate_a4_promotion": "PASS|FAIL|BLOCKED",
  "shadow_eligible": false,
  "active_eligible": false,
  "evidence_origin": "REAL_MODEL|SYNTHETIC|UNAVAILABLE"
}
```

### Run Directory Contract

Strict, immutable run-directory layout with 20 required files and 6 required
subdirectories. All files affecting scientific claims must appear in
`CHECKSUMS.sha256`.

### Promotion Manifest

Cryptographically binds a promoted policy to its qualification evidence via
SHA-256 digests of all gate results, policy artifact, training data, and
evaluation. An altered policy artifact invalidates the promotion record.

---

## Gate Changes

| Gate | v0.4.6 | v0.4.7 |
|---|---|---|
| Gate A0 | Evidence admissibility (24 sub-gates) | Same — wrapped into v0.4.7 GateResult |
| Gate A1 | Not separate | **NEW** — Routing headroom (oracle vs baseline/sham) |
| Gate A2 | Not separate | **NEW** — Candidate causal effectiveness (must beat both controls) |
| Gate A3 | Not separate | **NEW** — Robustness and replication (multi-seed) |
| Gate A4 | Not separate | **NEW** — Promotion eligibility (shadow/active) |
| `gate_a_status` | Ambiguous "PASS" | **REPLACED** — `legacy_gate_a_status` computed conservatively |

---

## Test Results

### Full Test Suite

```
pytest tests/ -q --timeout=60

983 passed, 1 skipped, 28 warnings in 27.66s
```

### v0.4.7 Specific Tests

```
pytest tests/unit/test_v047_gate_separation.py tests/unit/test_v047_artifact_integrity.py -v

54 passed in 0.37s
```

### Test Categories

| Category | Tests | Status |
|---|---|---|
| Gate separation | 8 | PASS |
| Statistical evaluation | 8 | PASS |
| Evidence enforcement | 7 | PASS |
| Safety evaluation | 4 | PASS |
| Version identity | 4 | PASS |
| Artifact integrity | 9 | PASS |
| Promotion binding | 7 | PASS |
| Serialization parity | 3 | PASS |
| Reporting | 5 | PASS |
| Existing tests | 929 | PASS |
| **Total** | **983 + 1 skipped** | **ALL PASS** |

---

## Real-Run Results

**Status: COMPLETED**

The v0.4.7 evaluation was run on a RunPod RTX 5090 (32GB VRAM) using the
existing real-model evidence from the previous v0.4.6 run (Qwen2.5-3B-Instruct,
80 tasks, 320 counterfactual outcomes, REAL_MODEL origin).

### Machine-Readable Verdict

```json
{
  "release": "2.15.11",
  "autolearn": "0.3.10",
  "qualification": "0.4.7",
  "protocol": "0.4.7",
  "gate_a0_admissibility": "PASS",
  "gate_a1_headroom": "PASS",
  "gate_a2_effectiveness": "FAIL",
  "gate_a3_robustness": "FAIL",
  "gate_a4_promotion": "FAIL",
  "shadow_eligible": false,
  "active_eligible": false,
  "evidence_origin": "REAL_MODEL",
  "legacy_gate_a_status": "FAIL"
}
```

### Gate Results

| Gate | Status | Key Finding |
|---|---|---|
| A0 (admissibility) | **PASS** | 24/24 sub-gates passed, REAL_MODEL evidence valid |
| A1 (routing headroom) | **PASS** | Oracle vs baseline LCB=0.227 > 0.02; Oracle vs sham LCB=2.69 > 0.02 |
| A2 (effectiveness) | **FAIL** | Candidate vs baseline: mean_delta=0.0, tie_rate=1.0 (candidate identical to baseline) |
| A3 (robustness) | **FAIL** | Single-seed run (blocked by design); action collapse detected |
| A4 (promotion) | **FAIL** | Not promotion eligible (A2 and A3 not PASS) |

### Gate A2 Detail

The candidate vs baseline comparison showed:
- **mean_delta = 0.0** — the candidate policy is identical to baseline
- **tie_rate = 1.0** — all 80 tasks have identical utility
- **LCB = 0.0** — does not exceed epsilon_0 = 0.01

The candidate vs sham comparison showed:
- **mean_delta = 3.13** — candidate clearly beats sham
- **LCB = 1.99** — exceeds epsilon_S = 0.01
- **passes = True**

**Scientific conclusion:** The candidate beats sham but NOT baseline. This is
correctly reported as FAIL. The simple heuristic candidate policy in
`run_local_gpu_scientific.py` produces the same action selections as the
baseline policy, resulting in zero delta. A real learned policy (trained via
`SupervisedRouterLearner` on counterfactual experience) would be needed to
demonstrate Gate A2 PASS.

### Gate A3 Detail

Gate A3 is BLOCKED by design for single-seed runs:
- Only 1 generation seed (requires ≥ 3)
- Only 1 learner seed (requires ≥ 3)
- Action-distribution collapse: max_single_action_share=1.0 > 0.98

### This is a scientifically honest failure

Per the spec: "A scientifically honest failure is a more valuable result than
another nominally successful but unsupported AutoLearn claim."

The v0.4.7 gate hierarchy correctly:
1. Passed Gate A0 (evidence is admissible)
2. Passed Gate A1 (routing headroom exists)
3. **Failed Gate A2** (candidate does not beat baseline — honestly reported)
4. **Failed Gate A3** (single-seed — blocked by design)
5. **Failed Gate A4** (not promotion eligible)
6. Set `legacy_gate_a_status = "FAIL"` (not "PASS" — the core fix)

---

## Unresolved Issues

1. **Candidate policy does not beat baseline**: The heuristic candidate policy
   in `run_local_gpu_scientific.py` produces identical selections to baseline
   (mean_delta=0.0). A real learned policy trained via `SupervisedRouterLearner`
   on counterfactual experience is needed to demonstrate Gate A2 PASS.

2. **Multi-seed replication (Gate A3)**: The current `run_local_gpu_scientific.py`
   generates evidence with a single generation seed. Gate A3 is BLOCKED for
   single-seed runs by design. A multi-seed runner needs to be implemented
   to achieve Gate A3 PASS.

3. **`qualify` CLI command**: The `qualify` command is a stub that returns
   `EXIT_RUNTIME_ERROR`. Evidence generation still uses
   `run_local_gpu_scientific.py`. The `evaluate` command is functional.

4. **Post-shadow validation**: Gate A4 sets `active_eligible=False` by design.
   Post-shadow validation infrastructure is not yet implemented.

5. **Historical code reorganization**: The spec suggests reorganizing
   `qualification/` into `current/v047/` and `archive/` directories. This was
   not done to avoid breaking imports. Compatibility shims would be needed.

---

## Reproduction Commands

```bash
# Run all tests
PYTHONPATH=src python -m pytest tests/ -q --timeout=60

# Run v0.4.7 tests only
PYTHONPATH=src python -m pytest tests/unit/test_v047_gate_separation.py tests/unit/test_v047_artifact_integrity.py -v

# Verify all v047 modules import
PYTHONPATH=src python -c "from qualification.autolearn_v04.v047 import gate_schema, config, gate_a0, gate_a1, gate_a2, gate_a3, gate_a4, statistics, family_evaluation, collapse_checks, safety_evaluation, evidence_enforcement, run_directory, artifact_dag, promotion_manifest, report, cli, orchestrator; print('OK')"

# Run v0.4.7 CLI
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli --version
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli audit --evidence-dir <dir>
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli evaluate --evidence-dir <dir>
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli report --evidence-dir <dir>
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli verify-checksums --evidence-dir <dir>
PYTHONPATH=src python -m qualification.autolearn_v04.v047.cli promotion-check --evidence-dir <dir>
```

---

## Final Verdict

```
Runtime integration:              PASS (existing tests)
Evidence admissibility:           PASS (Gate A0, 24/24 sub-gates)
Routing headroom:                 PASS (Gate A1, oracle exceeds baseline/sham)
Candidate causal effectiveness:   FAIL (Gate A2, candidate = baseline, delta=0.0)
Replication and robustness:       FAIL (Gate A3, single-seed, blocked by design)
Shadow promotion:                 NOT ELIGIBLE
Active promotion:                 NOT ELIGIBLE
Gate B:                           NOT RUN
```

**Scientific conclusion:** The v0.4.7 gate hierarchy correctly separated
evidence admissibility (Gate A0 PASS) from causal effectiveness (Gate A2 FAIL).
The real-model evidence from Qwen2.5-3B-Instruct is admissible and shows
routing headroom (Gate A1 PASS), but the heuristic candidate policy does not
beat the frozen baseline (mean_delta=0.0, tie_rate=1.0). The candidate does
beat the sham control (LCB=1.99 > 0.01), but Gate A2 requires beating BOTH
controls. This is a scientifically honest failure: the existing heuristic
candidate is not a learned policy and cannot demonstrate causal effectiveness.
A real `SupervisedRouterLearner`-trained policy on counterfactual experience
is needed to attempt Gate A2 PASS. The `legacy_gate_a_status` is correctly
"FAIL" — not "PASS" — which is the core fix this release delivers.
