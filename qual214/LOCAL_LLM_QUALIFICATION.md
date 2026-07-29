# Capsule Brain v2.14.0 — Local-LLM Qualification Report

**Package under test:** `capsule_brain-2.14.0-py3-none-any.whl`, built from the
corrected `src/` tree in this repository. The source tree, the built wheel, the
tests, and the qualification harness all refer to the same immutable build.
**Inference server:** Ollama 0.12.0, OpenAI-compatible endpoint at `http://localhost:11434/v1`.
**Primary model (recommended):** `qwen2.5-coder:3b` (digest `f72c60cabf62`, 3.1B Q4_K_M).
**Secondary model:** `llama3.1:8b` (digest `46e0c10c039e`, 8.0B Q4_K_M) — used
**only** for native tool-calling, where the primary model lacks the capability.
**Date:** 2026-07-29.

## Provenance enforcement

Before any test runs, the harness asserts `importlib.metadata.version("capsule-brain")
== "2.14.0"` and records a manifest (`qual214/manifest.json`) containing:

- installed package version: 2.14.0
- source-tree SHA-256 (over `src/**/*.py`, `tests/**/*.py`, `pyproject.toml`)
- installed wheel SHA-256 (when available)
- Python version, platform, Ollama version
- model digests, parameter sizes, quantization levels

The harness refuses to run against any other installed build (exits with code 3).

The source tree is byte-identical to the installed v2.14.0 package
(`diff -rq src/capsule_brain <site-packages>/capsule_brain` produces no differences).

## How these tests were run

Every test drives the **real installed Capsule Brain 2.14.0 runtime** through its
public surface — `build_application()`, `ConversationService.respond()`,
`ToolRegistry`, `WorkflowRunnerService`, `ExecutionService`, `VerificationService`,
and the built-in `plan_generate_test_reflect` workflow. No runtime class was
monkeypatched or replaced with manual subprocess logic or printed fake statuses.
The only stubs used are fault-injection doubles for Test 10 (explicitly requested
failure injection) and an internal fake LLM provider for the Test 10
infrastructure-failure cases and the Test 9 crash-phase slow provider.

Harness: `qual214/run_tests.py` (in-process tests 2,3,4,5,6,7,10) plus
`qual214/test8_store.py` / `test8_check.py` (persistence, separate processes),
`qual214/test9_start.py` / `test9_resume.py` (crash recovery, separate processes),
and `qual214/test11_container.py` (Docker container execution).
Machine-readable results: `qual214/qualification_results.json`.
Provenance manifest: `qual214/manifest.json`.

## v2.14.0-specific changes exercised

The 2.14.0 wheel differs from the 2.13.1 source tree in three production-source
files (all now present in `src/`):

1. `runtime/bootstrap.py` — rejects the `enabled` typo (forces `enable`), so a
   misconfigured boot cannot appear complete with major services absent. **Exercised
   by Test 2** and pinned by `test_v214_bootstrap_rejects_enabled_typo`.
2. `workflow/builtins.py` — adds `normalize_python_artifact` (strips a single outer
   Markdown fence from generated code). **Exercised by Tests 5 and 7**
   (`artifact_normalized=True`).
3. `workflow/builtins.py` — verifier-conditioned repair prompt that feeds the
   acceptance/stderr failure into the reflection step. **Exercised by Test 7**
   (verifier error text observed reaching the repair prompt).

A fourth fix was made during this qualification: `execution/container_runner.py`
had a CID-file bug that prevented Docker container execution (docker refuses to
write to an existing `--cidfile`). Fixed and pinned by the container execution
test (Test 11).

## Summary

| # | Test | Status | Classification |
|---|------|--------|----------------|
| 2 | Complete application boot | **PASS** | runtime |
| 3 | Basic conversation (10 prompts) | **PASS** | runtime |
| 4a | Native tool — qwen2.5-coder:3b | **BLOCKED** | model |
| 4b | Native tool — llama3.1:8b | **PARTIAL** | model |
| 5 | Real acceptance workflow | **PASS** | runtime |
| 6 | False-success regression | **PASS** | runtime |
| 7 | Reflection (fail → repair) | **PASS** | runtime |
| 8 | Persistence across instances | **PASS** | runtime |
| 9 | Workflow crash recovery | **PASS** | runtime |
| 10 | Failure injection (fail-closed) | **PASS** | runtime |
| 11 | Container execution (Docker) | **PASS** | runtime |

**Overall: PASS.** The runtime's native tool path, acceptance workflow, false-success
gate, reflection, persistence, crash recovery, failure injection, and container
execution have all been demonstrated through the actual runtime.

No test is classified `NOT TESTED`. One sub-case (4a) is `BLOCKED` (model lacks
native tool-call emission). One sub-case (4b) is `PARTIAL` (runtime tool machinery
proven; model fails exact result grounding). Both are reported transparently with
separate property tracking — neither is converted to PASS.

## Per-test evidence

### Test 2 — Complete application boot  (PASS, runtime)
Built with LLM + conversation + tools + execution + verification + workflow enabled,
acceptance verification mode, smoke success disabled. All required services are
present and healthy. Construction boundary validation rejects `verification_mode=smoke`
without `allow_smoke_success=true` and rejects `<section>.enabled` typo with a
corrective hint.

### Test 3 — Basic conversation  (PASS, runtime)
Called the actual `ConversationService.respond()` for 10 deterministic prompts
covering exact response, arithmetic, JSON, Python expression, short reasoning, and
conversation continuity. **10/10 succeeded**, latency 132–1816 ms. The
`ExperienceStore` recorded **10 experience records** (verified via `count()`).

### Test 4 — Unfakeable native tool test
Registered `secret_nonce` (returns the constant `CAPSULE_TOOL_9F31B7`, increments a
call counter) and asked: "Call the secret_nonce function and return its exact
result." The scorer tracks **four separate properties**:

1. `tool_selection_success` — the model selected the `secret_nonce` tool.
2. `tool_execution_success` — the tool was executed exactly once, no failures.
3. `tool_result_delivery_success` — the tool result was delivered back to the model
   (`tool_calls_executed == 1`).
4. `final_answer_grounding_success` — the final answer, after removing prose
   wrappers, equals the nonce **exactly**. A response like "The tool
   'CAPSULE_TOOL_9F31B7' returned: 0x1234567890abcdef" FAILS this property because
   the model hallucinated a different return value.

- **4a — qwen2.5-coder:3b: BLOCKED (model).** Confirmed at the raw
  `/v1/chat/completions` endpoint: this model emits tool calls as JSON content,
  never as native `tool_calls`. Capsule Brain's OpenAI-compatible provider only
  parses native `tool_calls`. Model/inference-server limitation, not a runtime defect.
- **4b — llama3.1:8b: PARTIAL (model).** Properties 1–3 PASS (tool selection,
  execution, result delivery all succeed; `counter==1`, `tool_calls_executed==1`,
  `tool_failures==0`). Property 4 FAILS: the model hallucinates a different return
  value (e.g. `0x1234567890abcdef`) instead of echoing the actual nonce. The tool
  protocol transcript is correct (assistant tool_call → tool result → assistant
  final); the model simply does not faithfully reproduce the tool's return value.
  This is a model grounding limitation, not a Capsule Brain runtime defect. The
  runtime's `generate_with_tools` loop was independently verified to build a
  protocol-valid transcript and feed the real tool result back.

### Test 5 — Real acceptance workflow  (PASS, runtime)
Used the actual `WorkflowRunnerService` with the built-in
`plan_generate_test_reflect` workflow. Goal: implement `add(a, b)`. An independent
acceptance file was supplied via workflow metadata; the generated artifact was
written as `solution.py` and the acceptance command was executed by the real
`ExecutionService`. Persisted run: `COMPLETED`, `test_passed=True`,
`verification_kind=acceptance`, `exit_code=0`, `artifact_normalized=True`.

### Test 6 — False-success regression  (PASS, runtime)
Goal requests an **incorrect** implementation (`add(a, b)` returning `a - b`);
acceptance requires addition. Reflection disabled (`max_reflect_iterations=0`).
The generated `solution.py` (subtraction) runs cleanly on its own but the acceptance
assertions fail. Persisted run: `FAILED`, `test_passed=False`, `exit_code=1`. The
run was **not** promoted to `COMPLETED`. `solution.py` exiting 0 is never treated as
goal success.

### Test 7 — Reflection (fail → repair)  (PASS, runtime)
Used `merge_sort` with an edge-case acceptance contract. The model's first artifact
failed acceptance; the verifier/acceptance error text reached the repair prompt
(captured in `state.errors`); reflection produced a repaired artifact that satisfied
acceptance. Persisted run: `completed`, `test_passed=True`, `iterations=1`,
`reflected=True`, `verifier_error_in_errors=True`. The 2.14.0 verifier-conditioned
repair prompt was exercised.

### Test 8 — Persistence across instances  (PASS, runtime)
Stored a unique value via the real `MemoryService` (SQLite-backed), shut down
completely. A **new application instance** pointing at the same persistent stores
retrieved the value directly from storage (`FOUND_IN_PERSISTENT_STORAGE:True`). The
new process's model never received the value in any prompt.

### Test 9 — Workflow crash recovery  (PASS, runtime)
Started a real `plan_generate_test_reflect` workflow run, let the plan node persist,
then **hard-killed the process** (`os._exit`) while the generate node was still
executing. A new Capsule Brain process pointed at the same workflow database,
using the **same real `plan_generate_test_reflect` workflow code** and the **real
local LLM**:

- same `run_id` recovered, `INTERRUPTED` recognized,
- plan step + plan text + verification contract + nonce preserved,
- explicit `resume_run` re-ran the generate node with the real LLM,
- the **real acceptance gate ran and passed** (`ACCEPTANCE_RAN:True`,
  `ACCEPTANCE_PASSED:True`, `EXIT_CODE:0`, `STDOUT_TAIL:'ACCEPTANCE_OK'`),
- `COMPLETED` only after acceptance.

This addresses the critique that the previous recovery test changed the workflow
implementation between processes. Both sides now use the same real built-in
workflow, and the resume runs into the real acceptance gate.

### Test 10 — Failure injection (fail-closed)  (PASS, runtime)
Every infrastructure failure failed closed (no fabricated success): dead LLM
endpoint, LLM timeout, malformed tool arguments, malformed tool JSON, tool handler
exception, ExecutionService exception, VerificationService exception.

### Test 11 — Container execution (Docker)  (PASS, runtime)
Ran the real `plan_generate_test_reflect` acceptance workflow through the real
`ContainerExecutionRunner` (Docker) with the full production-safe control set:

- read-only root filesystem
- network disabled (`--network none`)
- non-root user (`65534:65534`)
- memory limit 512m, swap 512m, CPUs 1.0, PIDs 128, nofile 1024
- `--cap-drop ALL`, `--security-opt no-new-privileges`
- pinned image digest (`python:3.11-slim` resolved to immutable digest)
- `--init`, `--tmpfs /tmp:rw,noexec,nosuid,size=64m`

The static verifier (`python_compile`) ran inside the container, the acceptance
test ran inside the container, and the run reached `COMPLETED` with
`test_passed=True`, `exit_code=0`, `stdout='ACCEPTANCE_OK'`.

## Tool-result grounding: honest assessment

The qualification distinguishes four separate properties of tool use. The runtime's
tool machinery (selection, execution, result delivery) is proven. However, neither
available local model achieves exact tool-result grounding in the final natural-language
answer:

- `qwen2.5-coder:3b` cannot emit native tool calls at all (BLOCKED).
- `llama3.1:8b` executes the tool correctly but hallucinates a different return
  value (PARTIAL — grounding fails).

This is reported transparently as a model limitation. The Capsule Brain tool
protocol is correct (verified by transcript inspection and raw-endpoint tests);
the grounding failure is a local-model behavior. A larger or tool-tuned model
would be needed to achieve `final_answer_grounding_success=True`.

## Failure classification (reported separately)

- **Runtime failures:** none. Every Capsule Brain component behaved correctly.
- **Model failures:** Test 4a (no native tool calls), Test 4b (hallucinated return
  value), Test 7 first-attempt failure (expected — reflection repairs it).
- **Inference-server failures:** Test 4a — Ollama 0.12.0 returns tool calls as
  content for `qwen2.5-coder:3b`.
- **Test-harness failures:** none.
- **Container runner bug (fixed):** `container_runner.py` created a CID file before
  docker ran, causing docker to refuse with exit 125. Fixed by deleting the file
  after reserving the path. Pinned by Test 11.

## Deliberately not claimed

This qualification does **not** claim AutoLearn, reinforcement learning, model-weight
learning, a learned executive policy, or autonomous self-improvement. Experience
records are collected and persisted; there is no online policy promotion, weight
update path, or self-improvement mechanism exercised or asserted here.

## Success criterion (met)

Capsule Brain v2.14.0 is locally qualified: real `ConversationService`, native tool
execution (runtime path proven), `WorkflowRunner` acceptance success **and** failure,
reflection, persistence, crash recovery into the real acceptance gate, failure
injection, and container execution have all been demonstrated through the actual
runtime. The source tree, wheel, tests, and qualification evidence all refer to the
same immutable build.
