# Capsule Brain v2.14.0 — Local-LLM Qualification Report

**Package under test:** `capsule_brain-2.14.0-py3-none-any.whl` (installed into a clean
Python 3.12 venv; `pip show capsule_brain` → `capsule-brain 2.14.0`).
**Inference server:** Ollama 0.12.0, OpenAI-compatible endpoint at `http://localhost:11434/v1`.
**Primary model (recommended):** `qwen2.5-coder:3b`.
**Secondary model:** `llama3.1:8b` — used **only** where the recommended model lacks a
required capability (native tool-call emission). This is documented as a model
limitation, not a Capsule Brain defect.
**Date:** 2026-07-29.

## How these tests were run

Every test drives the **real installed Capsule Brain 2.14.0 runtime** through its
public surface — `build_application()`, `ConversationService.respond()`,
`ToolRegistry`, `WorkflowRunnerService`, `ExecutionService`, `VerificationService`,
and the built-in `plan_generate_test_reflect` workflow. No runtime class was
monkeypatched or replaced with manual subprocess logic or printed fake statuses.
The only stubs used are fault-injection doubles for Test 10 (explicitly requested
failure injection) and an internal fake LLM provider for the Test 10
infrastructure-failure cases (a real network LLM cannot be made to deterministically
raise "daemon unreachable").

Harness: `qual214/run_tests.py` (in-process tests 2,3,4,5,6,7,10) plus
`qual214/test8_store.py` / `test8_check.py` and `qual214/test9_start.py` /
`test9_resume.py` (separate processes for persistence and crash recovery).
Machine-readable results: `qual214/qualification_results.json`.

## v2.14.0-specific changes exercised

The 2.14.0 wheel differs from the 2.13.1 source tree in exactly two files:
- `runtime/bootstrap.py` — rejects the `enabled` typo (forces `enable`), so a
  misconfigured boot cannot appear complete with major services absent. **Exercised
  by Test 2** (smoke-success-disabled + typo rejection).
- `workflow/builtins.py` — adds `normalize_python_artifact` (strips a single outer
  Markdown fence from generated code) and a verifier-conditioned repair prompt that
  feeds the acceptance/stderr failure into the reflection step. **Exercised by
  Tests 5 and 7** (artifact normalization observed; verifier error text observed
  reaching the repair prompt).

## Summary

| # | Test | Status | Classification |
|---|------|--------|----------------|
| 2 | Complete application boot | **PASS** | runtime |
| 3 | Basic conversation (10 prompts) | **PASS** | runtime |
| 4a | Native tool — qwen2.5-coder:3b | **BLOCKED** | model |
| 4b | Native tool — llama3.1:8b (runtime demonstration) | **PASS** | runtime |
| 5 | Real acceptance workflow | **PASS** | runtime |
| 6 | False-success regression | **PASS** | runtime |
| 7 | Reflection (fail → repair) | **PASS** | runtime |
| 8 | Persistence across instances | **PASS** | runtime |
| 9 | Workflow crash recovery | **PASS** | runtime |
| 10 | Failure injection (fail-closed) | **PASS** | runtime |

**Overall: PASS.** The success criterion is met: real `ConversationService`, native
tool execution, `WorkflowRunner` acceptance success **and** failure, persistence, and
crash recovery have all been demonstrated through the actual runtime.

No test is classified `NOT TESTED`. One sub-case (4a) is `BLOCKED` and reported as a
model/inference-server limitation with raw-endpoint evidence; it is not converted
into a PASS.

## Per-test evidence

### Test 2 — Complete application boot  (PASS, runtime)
Built with LLM + conversation + tools + execution + verification + workflow enabled,
acceptance verification mode, smoke success disabled. All required services are
present and healthy: `llm_gateway`, `tool_registry`, `conversation`,
`experience_store`, `execution`, `verification`, `workflow_runner`. Construction
boundary validation rejects `verification_mode=smoke` without
`allow_smoke_success=true`.

### Test 3 — Basic conversation  (PASS, runtime)
Called the actual `ConversationService.respond()` for 10 deterministic prompts
covering exact response, arithmetic, JSON, Python expression, short reasoning, and
conversation continuity (a shared conversation id across two turns).
Result: **10/10 succeeded**, latency 132–1816 ms, provider `openai`, model
`qwen2.5-coder:3b`. The `ExperienceStore` recorded **10 experience records** (one per
assistant response), each carrying `tool_calls_executed`, `tools_used`, and
`tool_failures` metadata.

### Test 4 — Unfakeable native tool test
Registered `secret_nonce` (returns the constant `CAPSULE_TOOL_9F31B7`, increments a
call counter) and asked: "Call the secret_nonce function and return its exact
result." PASS requires the nonce in the response **and** `counter==1` **and**
`tool_calls_executed==1` **and** `tools_used` contains `secret_nonce` **and**
`tool_failures==0`. A model merely claiming it called the tool is FAIL.

- **4a — qwen2.5-coder:3b: BLOCKED (model).** Before blaming the runtime, the raw
  `/v1/chat/completions` endpoint was tested with the same tool schema under
  `tool_choice` = `auto`, `required`, and a specific function. In every case the
  model emitted the tool call as **JSON content inside a Markdown fence**, never as a
  native `tool_calls` field. Capsule Brain's OpenAI-compatible provider only parses
  native `tool_calls`, so native execution cannot occur with this model. This is a
  model/inference-server limitation, not a Capsule Brain defect, and is recorded as
  BLOCKED (not FAIL, not PASS).
- **4b — llama3.1:8b: PASS (runtime).** Demonstrates the runtime's native tool path
  through the real `ConversationService`. The model emitted a native `tool_call`; the
  `ToolRegistry` executed it exactly once (`counter==1`, `tool_calls_executed==1`,
  `tools_used=["secret_nonce"]`, `tool_failures==0`); the tool result was fed back
  and the final answer contained the exact nonce `CAPSULE_TOOL_9F31B7`. The
  `generate_with_tools` loop was independently verified to build a protocol-valid
  transcript (assistant tool-call turn + matching `tool` role message) and to feed the
  real tool result back to the model.

### Test 5 — Real acceptance workflow  (PASS, runtime)
Used the actual `WorkflowRunnerService` with the built-in
`plan_generate_test_reflect` workflow. Goal: implement `add(a, b)`. An independent
acceptance file (`from solution import add; assert add(2,3)==5; assert add(-1,1)==0`)
was supplied via workflow metadata; the generated artifact was written as
`solution.py` beside it and the acceptance command was executed by the real
`ExecutionService` (host runner, sandboxed per-run directory).

Persisted run reports: `WorkflowStatus.COMPLETED`, `test_passed==True`,
`verification_kind=="acceptance"`, `verification_contract_present==True`,
`exit_code==0`. The 2.14.0 `normalize_python_artifact` path was exercised
(`artifact_normalized==True` — the model wrapped the code in a fence and the runtime
stripped the outer fence before verification).

### Test 6 — False-success regression  (PASS, runtime)
Goal requests an **incorrect** implementation (`add(a, b)` returning `a - b`); the
acceptance contract requires addition. Reflection was disabled
(`max_reflect_iterations=0`). The generated `solution.py` (subtraction) runs cleanly
on its own (exits 0), but the acceptance assertions fail.

Persisted run reports: `WorkflowStatus.FAILED`, `test_passed==False`,
`exit_code==1` (acceptance assertion error), `iterations==0`. The run was **not**
promoted to `COMPLETED`. This confirms `solution.py` exiting 0 is never treated as
goal success — the acceptance contract is authoritative.

### Test 7 — Reflection (fail → repair)  (PASS, runtime)
Used `merge_sort` with an edge-case acceptance contract (empty list, single
element, two-element, unsorted). The model's first artifact failed acceptance; the
verifier/acceptance error text reached the repair prompt (captured in
`state.errors`); reflection produced a repaired artifact that satisfied acceptance.

Persisted run reports: `status==completed`, `test_passed==True`,
`iterations==1` (one reflect→generate→test cycle), `reflected==True`,
`verifier_error_in_errors==True`. The 2.14.0 verifier-conditioned repair prompt was
exercised. (Local generation is stochastic; the harness allows up to 3 attempts and
reports each — the first attempt demonstrated fail-then-repair.)

### Test 8 — Persistence across instances  (PASS, runtime)
Stored a unique value `ORBIT-41-<nonce>` via the real `MemoryService` (SQLite-backed),
then shut the application down completely. A **new application instance** pointing at
the same persistent stores retrieved the value directly from storage
(`FOUND_IN_PERSISTENT_STORAGE:True`, `RECORD_COUNT:1`). The new process's model never
received the value in any prompt; it was retrieved from the persistent SQLite store,
not from model context.

### Test 9 — Workflow crash recovery  (PASS, runtime)
Started a real workflow run on the actual `WorkflowRunnerService`, let it persist the
first node, then **hard-killed the process** (`os._exit`) while the second node was
still executing (simulating a crash). A new Capsule Brain process pointed at the same
workflow database:

- same `run_id` recovered,
- `INTERRUPTED` recognized (stale `RUNNING` rows converted to `INTERRUPTED` on
  startup),
- previous steps preserved (`STEPS_PRESERVED:2`, `PREP_STEP_PRESERVED:True`),
- verification contract preserved in run metadata (`CONTRACT_PRESERVED:True`,
  `NONCE_PRESERVED:True`),
- safe continuation: explicit `resume_run` re-executed the interrupted node and
  completed (`STATUS_AFTER_RESUME:completed`, `STEPS_AFTER_RESUME:4`,
  `COMPLETED:True`).

### Test 10 — Failure injection (fail-closed)  (PASS, runtime)
Every infrastructure failure failed closed (no fabricated success):

| Injection | Outcome |
|---|---|
| Local LLM unavailable (dead endpoint) | `ConversationService` raised; no fake answer (`10a`) |
| LLM timeout (1 ms) | `LLMTimeoutError` raised (`10b`) |
| Malformed tool arguments (wrong type) | `ToolRegistry` returned `is_error`, no raise (`10c`) |
| Malformed tool JSON (parse error) | `ToolRegistry` returned `is_error` (`10c`) |
| Tool handler exception | `ToolRegistry` returned `is_error`, did not propagate (`10d`) |
| Execution infrastructure exception | workflow `FAILED`, `test_passed==False` (`10e`) |
| Verification service exception | workflow test node `test_passed==False`, `verification_kind=="static"` (`10f`) |

## Failure classification (reported separately)

- **Runtime failures:** none. Every Capsule Brain component behaved correctly.
- **Model failures:** Test 4a — `qwen2.5-coder:3b` does not emit native `tool_calls`
  via Ollama (BLOCKED). Test 7's `merge_sort` first-attempt failure is expected model
  behavior, not a defect. No runtime defect was caused by model behavior.
- **Inference-server failures:** Test 4a — Ollama 0.12.0 returns the tool call as
  content rather than in the `tool_calls` field for `qwen2.5-coder:3b`. Confirmed at
  the raw endpoint.
- **Test-harness failures:** none after the harness was corrected (initial harness
  bugs — `asyncio.run` inside a running loop, wrong `ExperienceStore` method name,
  and a goal the model solved on the first attempt — were fixed; they were never
  runtime defects).

## Deliberately not claimed

This qualification does **not** claim AutoLearn, reinforcement learning, model-weight
learning, a learned executive policy, or autonomous self-improvement. Experience
records are collected and persisted; there is no online policy promotion, weight
update path, or self-improvement mechanism exercised or asserted here.

## Success criterion (met)

Capsule Brain v2.14.0 is locally qualified: real `ConversationService`, native tool
execution, `WorkflowRunner` acceptance success **and** failure, persistence, and
crash recovery have all been demonstrated through the actual runtime rather than
substitute test code. The three highest-priority tests — unfakeable tool execution,
real acceptance workflow, and crash recovery — all pass.
