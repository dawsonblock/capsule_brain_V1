# Capsule Brain v2.13.0 — Verification Boundary & Tool-Path Correction

v2.13.0 is a correction release focused on evidence integrity. It does not add a new cognitive architecture. It fixes unsafe success semantics and closes a missing integration path discovered during the v2.12.1 audit.

## Corrections

### Construction-boundary configuration validation

`build_application()` now validates direct dictionary configuration before constructing the application. This closes the path where callers could bypass the host-execution acknowledgement enforced by `load_config()`.

### Acceptance verification is the workflow default

The default plan/generate/test/reflect workflow no longer treats a clean process exit as evidence that a goal was solved.

The workflow now expects request metadata containing an external verification contract. Generated Python is materialized as `solution.py` in a per-run sandbox directory alongside caller-supplied verification files. The supplied command is executed through `ExecutionService`; only that result can promote the run.

Missing or malformed acceptance evidence fails closed.

### Explicit legacy smoke mode

The previous execute-generated-code behavior remains available for smoke diagnostics only. It requires both:

```yaml
verification_mode: smoke
allow_smoke_success: true
```

This prevents infrastructure-level execution success from being mislabeled as semantic verification by default.

### VerificationService integration

When the shared verification service is available, the workflow runs Python/static execution-backed verification before acceptance. Result IDs and statuses are persisted in workflow state for auditability.

### Normal-conversation tool loop

`ConversationService` can now opt into the already-existing `LLMGateway.generate_with_tools()` loop. Registered tools are available to normal conversations when `conversation.use_tools=true`; no-tools behavior remains ordinary generation.

Tool execution counts, names, and failure counts are written into `ExperienceRecord.metadata`, providing better future policy-learning data.

## Deliberately not claimed

v2.13.0 is not a self-learning release. Experience collection is improved, but there is still no learned executive policy, weight update path, causal world model, or online policy promotion mechanism.
