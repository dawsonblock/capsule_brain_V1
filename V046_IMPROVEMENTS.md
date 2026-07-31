# Capsule Brain v2.15.10 — v0.4.6 Improvement Pass

**Date:** 2025-07-31
**Package:** Capsule Brain 2.15.10 / AutoLearn 0.3.9 / Qualification 0.4.6
**Scope:** Code-quality, performance, and robustness improvements applied after the initial Gate A0 qualification pass.

---

## 1. Executive Summary

After the 3B model passed all 24 Gate A0 sub-gates (documented in
`V046_GATE_A0_RESULTS.md`), a three-axis audit was performed across the v0.4.6
codebase:

1. **Code quality** — dead code, hardcoded values, error handling, type hints, duplication
2. **Correctness & robustness** — evidence generation, orchestrator safety, test gaps
3. **Performance & architecture** — GPU executor, evidence pipeline, orchestrator

Every confirmed finding was applied. All 929 unit tests pass (1 pre-existing
skip). The changes are spread across 6 commits on `main`:

| Commit | Summary |
|---|---|
| `ea06cf2` | Apply all improvement plan fixes (9 items across 3 subagents) |
| `1eb3395` | Fix CLI `--batch-size` default (1 -> 8) and reduce VRAM print noise |
| `02624bb` | Apply code quality and performance improvements |
| `3a054ee` | Fix critical `verifier_version` bug and harden None propagation |

---

## 2. Critical Fix: `verifier_version` in Counterfactual Outcomes

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py` (line 330)

**Bug:** Counterfactual outcome rows were setting `verifier_version` to the
function parameter value (`"scientific_v1"`) instead of the family-specific
version returned by `_verifier_for_family()` (e.g. `"1.0"`).

**Impact:** All 320 counterfactual rows would carry the wrong verifier version,
causing the A0.14 verifier registry audit to fail on the next real-model run.
The previous run passed only because the audit was tolerant of the mismatch;
the bug would have surfaced on stricter enforcement.

**Fix:**
```python
# Before
"verifier_version": verifier_version,   # function parameter

# After
"verifier_version": v_ver,              # family-specific version
```

---

## 3. Performance Improvements

### 3.1 O(n^2) -> O(1) Task Lookup

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`

The pattern `next((t for t in task_dicts if t["task_id"] == o["task_id"]), None)`
was used 3 times inside loops over outcomes. With 80 tasks x 4 actions = 320
outcomes, this is O(320 x 80) = 25,600 comparisons per loop, 76,800 total.

Replaced with a pre-built dict lookup:
```python
task_dict_lookup = {t["task_id"]: t for t in task_dicts}
# ...
task_dict = task_dict_lookup.get(o["task_id"])
```

### 3.2 KV Cache Clearing Frequency

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

`torch.cuda.empty_cache()` was called after every single generation and every
batch. This is an expensive synchronization operation that forces a GPU
device-wide stall.

| Mode | Before | After |
|---|---|---|
| Sequential | Every prompt | Every 40 prompts |
| Batched | Every batch | Every 10 batches |

The `past_key_values` reset still happens every call (it's cheap); only the
expensive `empty_cache()` was throttled.

### 3.3 `low_cpu_mem_usage=True`

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

Added `low_cpu_mem_usage=True` to the model loading kwargs. This uses the
accelerated `init_empty_weights` path, reducing peak CPU memory during model
loading from ~2x model size to ~1x.

### 3.4 `torch.compile()` Warmup

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

`torch.compile(mode="reduce-overhead")` has a one-time compilation cost that
makes the first generation significantly slower. Added a warmup generation
(`self.generate("Hello", max_new_tokens=1)`) immediately after compilation so
the first real batch isn't penalized.

### 3.5 CLI Batch Size Default

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`

The CLI argparse `--batch-size` default was `1` (sequential) even though the
function default was `8`. This meant running the CLI without `--batch-size 8`
silently used the slow sequential path. Fixed the CLI default to `8`.

### 3.6 VRAM Print Frequency

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

VRAM monitoring was printing after every prompt, flooding the log with 320
lines. Reduced to every 40 prompts (sequential) or every 5 batches (batched).

---

## 4. Code Quality Improvements

### 4.1 Extracted Duplicate Results Dict Construction

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`

The `baseline_results`, `candidate_results`, `sham_results`, and
`oracle_results` dicts were constructed with 4 copies of the same 8-field
pattern. Extracted into a single helper:

```python
def _build_results_dict(policy_id: str, task_rows: list[dict], n_tasks: int) -> dict:
    n_success = sum(1 for r in task_rows if r.get("success", False))
    mean_util = sum(r.get("selected_utility", 0.0) for r in task_rows) / max(1, len(task_rows))
    return {
        "schema_version": "results/1",
        "policy_id": policy_id,
        "n_tasks": n_tasks,
        "n_success": n_success,
        "mean_utility": round(mean_util, 6),
        "verified_success_rate": round(n_success / max(1, len(task_rows)), 6),
        "task_rows": task_rows,
    }
```

### 4.2 Removed Unused Imports

| File | Removed imports |
|---|---|
| `run_local_gpu_scientific.py` | `hashlib`, `_build_action_prompt` |
| `local_gpu_executor.py` | `re` |
| `v046/gate_a0_audit.py` | `AuditResult`, `make_pass`, `make_fail`, `make_blocked`, `make_not_applicable`, `aggregate_status`, `EvidenceOrigin` |

### 4.3 Specific Exception Handling

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

Replaced 3 bare `except Exception:` clauses with specific exception tuples:

| Location | Before | After |
|---|---|---|
| Model revision recording | `except Exception:` | `except (AttributeError, TypeError, KeyError):` |
| Chat template application | `except Exception:` | `except (ValueError, TypeError, AttributeError):` |
| Batch tokenization | `except Exception:` | `except (ValueError, TypeError, AttributeError):` |

### 4.4 JSON Error Handling in Orchestrator

**File:** `qualification/autolearn_v04/v046/orchestrator.py`

`_load_json` and `_load_jsonl` previously crashed with opaque tracebacks on
malformed JSON. Now they raise informative `ValueError` messages:

```python
# _load_json
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in {path}: {e}") from e

# _load_jsonl
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in {path} at line: {line[:80]}...: {e}") from e
```

### 4.5 Improved Type Hints

**File:** `qualification/autolearn_v04/v046/orchestrator.py`

| Function | Before | After |
|---|---|---|
| `_load_json` | `-> dict` | `-> dict[str, Any]` |
| `_load_jsonl` | `-> list[dict]` | `-> list[dict[str, Any]]` |
| `run_all_v046_diagnostics` | `-> dict` | `-> dict[str, Any]` |

---

## 5. Robustness Improvements

### 5.1 None Propagation Guards

**File:** `qualification/autolearn_v04/v046/orchestrator.py`

Three audit functions could return `None` in edge cases, which would crash
downstream `.get()` calls:

| Location | Guard |
|---|---|
| `evidence_origin` (Stage 2) | `audit_evidence_origin(...) or {}` |
| `provider_manifest` (Stage 21) | `_pm = provider_manifest or {}` |
| `gate_a0` (Stage 22) | `(gate_a0 or {}).get("status", "BLOCKED")` |

### 5.2 Gate A Evaluation Error Capture

**File:** `qualification/autolearn_v04/v046/orchestrator.py`

When `--evaluate-gate-a` is used and the Gate A evaluation crashes, the error
was previously only printed to stdout. Now it's also written to
`gate_a_evaluation.json`:

```python
except Exception as e:
    print(f"  Gate A evaluation failed: {e}")
    gate_a_status = "BLOCKED"
    _write("gate_a_evaluation.json", {
        "status": "BLOCKED",
        "error": str(e),
        "reason": "Gate A evaluation crashed",
    })
```

---

## 6. Features Added

### 6.1 `--evaluate-gate-a` CLI Flag

**Files:** `qualification/autolearn_v04/v046/cli.py`, `v046/orchestrator.py`

Added a CLI flag to run the full Gate A evaluation after Gate A0 passes with
real-model evidence. This enables the scale-up decision path (14B pilot /
full 14B run) without a separate invocation.

```bash
python -m qualification.autolearn_v04.v046.cli analyze \
  --evidence-dir /workspace/scientific_evidence_3b \
  --output-dir qualification/autolearn_v04/artifacts/v046_real \
  --run-id v046_real_3b_009 \
  --evaluate-gate-a
```

### 6.2 End-to-End Integration Test

**File:** `tests/unit/test_v046_evidence_repair.py`

Added `TestV046EndToEndPipeline.test_v046_end_to_end_pipeline` — generates a
complete minimal synthetic evidence package (8 tasks, 32 counterfactual
outcomes, 16 experience rows, all required files), runs the full v0.4.6
orchestrator, and asserts:

- Analysis manifest exists with correct origin (SYNTHETIC) and eligibility (PASS)
- Scale decision is NOT_APPLICABLE for synthetic evidence
- Gate A0 is not BLOCKED (0 blocked sub-gates)
- All 7 scientific sub-gates are NOT_APPLICABLE
- All 16 structural sub-gates are PASS
- All 30 required artifacts are produced

Runs in ~0.2s with minimal data.

### 6.3 Configurable Verifier Version

**File:** `qualification/autolearn_v04/run_local_gpu_scientific.py`

The verifier version is now a parameter (`verifier_version: str = "scientific_v1"`)
instead of being hardcoded, allowing different verifier versions to be used
without code changes.

### 6.4 tqdm Progress Bars

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

Replaced manual progress prints with `tqdm` progress bars for both sequential
and batched generation, providing ETA, rate, and elapsed time.

### 6.5 GPU VRAM Monitoring

**File:** `qualification/autolearn_v04/local_gpu_executor.py`

Added periodic VRAM monitoring during generation, reporting
`torch.cuda.memory_allocated()` at intervals (every 40 prompts / 5 batches).

---

## 7. Improvement Plan Items Applied

All 9 items from the improvement plan were applied:

| # | Item | Status |
|---|---|---|
| 1a | Fix split_manifest schema mismatch (add task_ids to splits) | Done |
| 1b | Fix batched generation default (1 -> 8) | Done |
| 2a | Improve evidence weight computation (populate quality fields on action rows) | Done |
| 2b | Fix artifact lineage display (n_artifacts_valid field) | Done |
| 3a | Add v0.4.6 end-to-end integration test | Done |
| 3b | Wire Gate A evaluation for scale-up path (--evaluate-gate-a flag) | Done |
| 4a | Add GPU VRAM monitoring during generation | Done |
| 4b | Replace progress prints with tqdm | Done |
| 4c | Make verifier version configurable | Done |

---

## 8. Test Results

```
929 passed, 1 skipped, 28 warnings in 26.34s
```

The 1 skip is pre-existing (unrelated to v0.4.6). The 76 v0.4.6-specific tests
all pass, including the new end-to-end integration test.

---

## 9. Audit Findings Considered but Not Applied

The following findings from the audit were evaluated but deliberately not
applied to avoid unnecessary churn:

| Finding | Reason |
|---|---|
| Replace 29 hardcoded status strings with `AuditStatus.PASS.value` etc. | The enum values are identical to the strings (`"PASS"`, `"FAIL"`, `"BLOCKED"`), so this is purely cosmetic churn with regression risk. |
| Parallelize independent orchestrator stages | The 25-stage pipeline is sequential but each stage is fast (<1s on real evidence). Parallelization would add complexity for negligible gain. |
| Cache source tree hash | The hash is computed once per run and takes <1s. Caching would add invalidation complexity for no measurable benefit. |
| Make `output_scores=True` configurable | The scores are required for Gate B logprob extraction. Disabling would break Gate B. |
| Exclude `run_id` from config digest | The `run_id` is intentionally part of the identity to distinguish runs. |
