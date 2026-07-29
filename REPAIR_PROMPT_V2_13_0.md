# Capsule Brain v2.12.1 -> v2.13.0 Correction Build Prompt

You are repairing and hardening the Capsule Brain v2 codebase. Treat this as a production engineering task, not a speculative rewrite. Preserve working architecture, persistence schemas, event contracts, and existing behavior unless a behavior is explicitly identified below as unsafe or false.

## Mission

Produce a corrected v2.13.0 release that closes the two highest-severity defects in v2.12.1 and finishes the missing normal-conversation tool integration without destabilizing the runtime.

The two critical defects are:

1. `build_application(cfg)` can bypass configuration validation when callers pass a Python dictionary directly instead of using `load_config()`.
2. The default autonomous coding workflow equates `ExecutionResult.passed` (`exit_code == 0 && !timed_out`) with successful completion of the user's goal. A program containing only `pass` can therefore be promoted as a successful solution.

The integration defect is:

3. `LLMGateway.generate_with_tools()` and `ToolRegistry` are implemented and tested, but `ConversationService.respond()` always calls `llm.generate()`, so normal conversations cannot use registered tools.

Do not add fake intelligence, unverified learning claims, automatic weight updates, or a new framework. Fix the system that exists.

---

## Non-negotiable system invariants

### Invariant A — Configuration safety is enforced at the construction boundary

Every application constructed by `build_application()` must be validated regardless of whether configuration originated from YAML, a test, an API, an embedding host, or a Python dict.

Host execution must never be enabled unless all of the following are true:

- `execution.enable == true`
- `execution.runner == "host"`
- `execution.unsafe_allow_host_execution == true`

If the acknowledgement is missing, application construction must fail before services are registered or started.

Preserve compatibility with callers/tests that historically treated invalid programmatic service dependency configuration as `RuntimeError` and YAML validation as `ValueError`. A dedicated configuration exception may inherit from both if useful.

### Invariant B — Execution success is not semantic verification

Never use this implication:

`exit_code == 0 -> user goal solved`

A generated artifact may complete a workflow only after an explicit verification contract passes.

The safe default workflow mode must be `acceptance`.

The workflow request metadata contract is:

```python
{
    "verification": {
        "files": {
            "acceptance.py": "...",
            "test_solution.py": "...",
        },
        "command": ["python", "acceptance.py"],
    }
}
```

The workflow must:

1. create a run-specific directory under `execution.policy.cwd_root`;
2. write generated code as `solution.py`;
3. write verification files only inside that run directory;
4. reject absolute paths and traversal attempts;
5. execute the supplied command through `ExecutionService`, never directly through `subprocess`;
6. set `test_passed = true` only when the verification command passes;
7. preserve stdout/stderr/exit code in workflow state;
8. clean temporary run artifacts;
9. fail closed when the verification contract is absent, malformed, blocked by policy, times out, or errors;
10. feed failure output into the existing reflection loop.

### Invariant C — Legacy smoke testing cannot masquerade as verification

Legacy behavior that simply runs generated Python may remain for diagnostics, but only as:

`verification_mode = "smoke"`

and only when:

`allow_smoke_success = true`

Application construction or workflow construction must reject smoke promotion unless the unsafe acknowledgement is explicit.

### Invariant D — Shared verification runs before acceptance when available

When `VerificationService` is running, invoke it on generated Python before the acceptance command using metadata such as:

```python
{
    "content_type": "python",
    "goal": state.goal,
    "workflow_run_id": state.run_id,
}
```

A verifier `FAIL` or verifier infrastructure error must block promotion.

Store the verification result ID/status in `WorkflowState.extra` for auditability.

### Invariant E — Normal conversation can use the existing native tool loop

Extend `ConversationService` with optional `tool_registry` integration.

Configuration:

```yaml
conversation:
  use_tools: false
  max_tool_iterations: 4
```

When `use_tools` is true and the registry contains tools:

- call `LLMGateway.generate_with_tools()`;
- preserve existing route/model/system prompt behavior;
- respect `ToolRegistry` permission, side-effect, timeout, JSON-schema and result-size policy;
- fall back to ordinary `generate()` when no tools are registered;
- do not silently fabricate tool results;
- keep tool use opt-in because not every model route has the `tools` capability.

Record into the `ExperienceRecord.metadata`:

- `tool_calls_executed`
- `tools_used`
- `tool_failures`

Expose tool configuration/registered tool names in `ConversationService.health()`.

---

## Required code changes

### 1. `src/capsule_brain/runtime/bootstrap.py`

- Introduce a configuration exception if needed for compatibility.
- Call `_validate_config(cfg)` immediately inside `build_application()` before constructing `CapsuleApplication`.
- Validate `workflow.verification_mode` is one of `acceptance` or `smoke`.
- Reject enabled smoke mode unless `workflow.allow_smoke_success == true`.
- Keep a `tool_registry` reference outside the local LLM initialization block.
- Pass `tool_registry` into `ConversationService`.
- Add `tool_registry` to conversation service dependencies when present.
- Pass `verification_mode` and `allow_smoke_success` into the default workflow builder.

### 2. `src/capsule_brain/workflow/runner.py`

- Persist workflow request metadata into state only when metadata exists, for example:

```python
state.extra["request_metadata"] = run_metadata
```

This allows workflow nodes to consume verification contracts after restart/resume without coupling node actions to `WorkflowRun` internals.

### 3. `src/capsule_brain/workflow/builtins.py`

Extend:

```python
build_plan_generate_test_reflect_workflow(...)
```

with:

```python
verification_mode: str = "acceptance"
allow_smoke_success: bool = False
```

Implement separate acceptance and smoke verification paths.

Acceptance is the default and is the only mode that should be described as verified goal completion.

Do not generate acceptance tests from the same LLM and then treat those tests as authoritative. Caller-supplied or independently produced acceptance assets are the evidence boundary.

Use per-run directories to prevent concurrent workflow collisions.

### 4. `src/capsule_brain/conversation/service.py`

- accept `tool_registry`;
- add `use_tools` and `max_tool_iterations` config;
- route normal conversation through `generate_with_tools()` when enabled and tools exist;
- preserve ordinary generation otherwise;
- record tool-use telemetry in experience metadata;
- expose tool-related health details.

### 5. `configs/v2_runtime.yaml`

Document:

```yaml
conversation:
  use_tools: false
  max_tool_iterations: 4

workflow:
  verification_mode: acceptance
  allow_smoke_success: false
```

Include an example verification metadata contract and explicitly state that smoke mode is not semantic verification.

### 6. Versioning/documentation

- bump package version to `2.13.0`;
- preserve v2.12 milestone history;
- add a v2.13 correction milestone document;
- update validation results only after actually executing tests/build gates.

---

## Mandatory regression tests

Add tests proving all of the following:

### Config validation

1. Direct `build_application({...})` with enabled host runner and no unsafe acknowledgement raises before startup.
2. YAML config validation still rejects the same configuration.
3. Smoke workflow configuration requires explicit acknowledgement.

### False-success elimination

4. Generated code equal to `pass` does not complete the default workflow when no acceptance verification contract is supplied.
5. Missing acceptance contract is explicitly represented in `state.stderr` and `state.extra`.
6. A correct `solution.py` plus independent acceptance script completes successfully.
7. A failing acceptance script prevents completion even when `solution.py` itself exits 0.
8. Verification file path traversal is rejected.
9. Execution/verification infrastructure failure fails closed.

### Tool integration

10. A normal `ConversationService.respond()` call with `use_tools=true` and a registered tool executes the tool loop and incorporates the tool result into the final response.
11. With no registered tools, conversation still uses ordinary generation.
12. Tool-use counts/names/errors are persisted in experience metadata.

### Existing regressions

All pre-existing tests must continue to pass except where an unsafe old expectation is intentionally replaced by an explicitly acknowledged smoke-mode test.

---

## Release gates

Run, in order:

```bash
python -m compileall -q src tests
pytest -q
PYTHONPATH=src python demo.py
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Then install/import the wheel in an isolated target and confirm at minimum:

```python
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.llm.tools import ToolRegistry
```

Do not write `PASS` into validation documentation unless the command was actually executed successfully.

---

## Explicitly out of scope for this correction build

Do not claim or fake any of the following:

- learned executive policy;
- reinforcement learning;
- automatic LoRA/weight updates;
- world-model learning;
- causal memory;
- true memory consolidation;
- autonomous goal-to-workflow dispatch without a verification contract;
- semantic correctness for arbitrary natural-language goals without an independent verifier.

Those are next-stage research/architecture tasks, not repair work.

---

## Next-stage architecture after v2.13.0

Once this correction release is green, the next major feature should be a verifier-driven policy layer over the existing experience records.

Target decision space:

```text
state
  -> answer directly
  -> retrieve memory
  -> call tool
  -> symbolic executor
  -> launch workflow
  -> request verification
  -> reflect
  -> ask operator
```

Collect verified outcome/latency/cost/tool/route data first. Train or update candidate policies offline. Promote only through held-out evaluation gates. Never allow an online learner to overwrite the active policy simply because it observed a single successful trajectory.

The architectural rule remains:

**Models propose. External evidence promotes.**
