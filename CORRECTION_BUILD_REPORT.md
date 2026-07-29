# Capsule Brain v2.13.0 — Correction Build Report

## Scope

This release repairs the highest-impact correctness/safety defects identified in the v2.12.1 audit without rewriting the runtime.

## Implemented corrections

1. **Construction-boundary config validation**
   - `build_application()` now invokes `_validate_config()` before constructing services.
   - Direct Python-dict configuration can no longer bypass the host-execution safety acknowledgement.
   - Added a compatibility `ConfigurationError(ValueError, RuntimeError)`.

2. **False-success workflow elimination**
   - Default workflow verification mode is now `acceptance`.
   - A clean process exit from generated code is no longer sufficient for workflow completion.
   - Workflow requests may carry an external verification contract containing files and a command.
   - Generated code is exposed as `solution.py` inside a unique per-run sandbox directory.
   - Missing/malformed acceptance evidence fails closed.
   - Verification failures feed the existing reflection loop.

3. **Explicit legacy smoke mode**
   - Old run-generated-code behavior is retained only as `verification_mode="smoke"`.
   - Promotion requires explicit `allow_smoke_success=True` acknowledgement.

4. **Shared verifier integration**
   - When `VerificationService` is running, generated Python is checked before acceptance execution.
   - Verification result ID/status are persisted in workflow state.

5. **Workflow request metadata persistence**
   - Run metadata is mirrored into `WorkflowState.extra["request_metadata"]` when present.
   - Verification contracts therefore survive persistence/resume boundaries.

6. **Normal-conversation native tool integration**
   - `ConversationService` now supports opt-in `LLMGateway.generate_with_tools()` execution.
   - Config: `conversation.use_tools`, `conversation.max_tool_iterations`.
   - Empty registries preserve normal generation behavior.
   - Tool use is reported in service health.

7. **Experience telemetry improvement**
   - `ExperienceRecord.metadata` now captures tool call count, tool names, and tool failures.
   - This improves the evidence available for a future verifier-driven routing/policy learner.

8. **Configuration/docs/versioning**
   - Package version bumped to `2.13.0`.
   - Runtime config documents safe acceptance verification and opt-in tool use.
   - Added v2.13 milestone, correction prompt, validation report, and preserved v2.12.1 validation history.

## New regression coverage

Added five focused v2.13 correction tests covering:

- direct `build_application` host-execution validation;
- smoke-mode acknowledgement;
- rejection of false success without acceptance evidence;
- successful external acceptance verification;
- native tool execution from normal conversation.

Full suite result: **176 passed**.

## Validated release gates

- `python -m compileall -q src tests` — PASS
- `pytest -q` — PASS, 176 tests
- `PYTHONPATH=src python demo.py` — PASS
- wheel build — PASS
- isolated wheel install/import — PASS

## Qualification boundary

No Docker/Podman executable was available in the build environment. The new acceptance path was therefore validated through the host `ExecutionRunner` under an explicit test policy. Live container qualification remains required before claiming deployment-level validation of the v2.13 acceptance path.

## What this release still is not

This is not a learned-policy or continual-learning release. Capsule Brain now has a stronger evidence boundary and better experience telemetry, but it still does not automatically update a learned executive policy, model weights, LoRA adapters, or a world model.
