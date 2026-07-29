# Capsule Brain v2.12.1 — Milestone 12 Architecture Upgrades & Hardening

Milestone 12 adds semantic memory, native LLM tool calling, OpenTelemetry
instrumentation, adaptive reflection, bounded execution workers, and a
persistent workflow/state-machine layer. The v2.12.1 hardening pass closes the
safety and integration defects found during a fresh codebase audit.

The governing invariant is fail-closed autonomy: missing execution, failed
verification, denied approval, cancellation, malformed tool input, or exhausted
repair attempts can never be promoted into success.

## 1. Semantic memory

- `EmbeddingProvider` has stable vector-space fingerprints.
- `HashEmbeddingProvider` remains a deterministic, zero-dependency bootstrap
  provider. It is not represented as a production semantic model.
- `OpenAICompatibleEmbeddingProvider` supports real `/embeddings` endpoints
  with explicit model, dimension, API base, API-key environment variable, and
  timeout configuration.
- Unknown provider names fail configuration instead of silently falling back to
  hash embeddings.
- `memory_embeddings` stores `dimension` and `fingerprint`; existing databases
  are migrated in place.
- Queries only compare vectors from the current fingerprint, preventing mixed
  embedding spaces after a provider/model change.
- `reindex_all()` provides an explicit migration path after embedding upgrades.
- `sqlite-vec` is optional. Native KNN is used only when its dimension table is
  fully backfilled and no fingerprint/archive filtering could corrupt top-K
  semantics. Otherwise the authoritative serialized-vector path is used.
- Python cosine scoring runs outside the repository lock and outside the event
  loop via `asyncio.to_thread`.
- Conversation semantic retrieval occurs before the current operator turn is
  persisted, so the query cannot consume one of its own top-K slots.
- Auto-index failures are counted, logged, and reflected as DEGRADED memory
  health while preserving the primary memory write.

Primary files: `memory/embeddings.py`, `memory/repository.py`,
`memory/sqlite_repository.py`, `memory/service.py`, `conversation/service.py`.

## 2. Native tool/function calling

- `ToolSpec`, `ToolCall`, and `ToolResult` define the provider-independent tool
  contract.
- `LLMRequest.messages` carries the canonical multi-turn transcript.
- OpenAI-compatible continuation preserves the required ordering:
  `system/user -> assistant(tool_calls) -> tool result -> assistant`.
- Provider `tool_choice` respects the request policy rather than being forced
  to `auto`.
- Malformed tool JSON is preserved as an explicit parse error; it is never
  converted into an empty argument object.
- Tool schemas are checked with JSON Schema Draft 2020-12 at registration and
  arguments are fully validated before handler dispatch.
- Registry policy supports permission classes, side-effect blocking, handler
  timeouts, and bounded result sizes.
- On the final permitted model iteration, pending tool calls are returned but
  are not executed because there is no subsequent model turn that could consume
  their results. This prevents unconsumed side effects.
- The shipped OpenAI model capability set includes `tools`; side-effecting tools
  remain disabled by default.

Primary files: `llm/tools.py`, `llm/models.py`, `llm/gateway.py`,
`llm/providers/openai_compatible.py`.

## 3. OpenTelemetry and event propagation

- Normal instrumentation uses `start_as_current_span`, so nested operations
  form real OpenTelemetry parent/child traces.
- W3C propagation carriers can be extracted when a span starts and injected
  into `EventEnvelope.trace_context` for downstream work.
- Correlation IDs remain searchable span attributes but are not used as a
  substitute for trace parentage.
- When `tracing.otlp_http_endpoint` is supplied, bootstrap can create an SDK
  provider with an OTLP/HTTP exporter. Without an endpoint, the configured
  global/provider behavior is retained.
- Redis bridge serialization preserves `event_id`, `correlation_id`,
  `created_at`, and `trace_context` across process boundaries.
- Redis is a lazy optional dependency: importing the bridge does not break a
  base installation; starting the bridge without the `redis` extra fails with
  a clear runtime error.

Primary files: `observability/tracing.py`, `events/models.py`,
`events/local_bus.py`, `events/redis_bridge.py`.

## 4. Adaptive reflection with authoritative verification

- `CodeSyntaxStrategy` retains deterministic `compile()` closure for syntax
  repairs.
- `PytestFailureStrategy` no longer asks an LLM to certify its own patch. It
  performs one focused repair call and then reruns `VerificationService` with
  event publication suppressed to avoid recursive reflection.
- A pytest repair is resolved only when an actual pytest verifier returns PASS.
- Verification failure events contain complete serialized checks and detailed
  stdout/stderr/diff metadata instead of only aggregate summaries.
- Reflection seeds receive the actual failing-check context.
- Reflection timeout handling preserves the latest generated revision instead
  of reverting to the original seed.

Primary files: `reflection/strategies.py`, `reflection/service.py`,
`verification/service.py`.

## 5. Verification and execution isolation

- Execution-backed verifiers create unique temporary artifacts per request,
  pass container-safe relative filenames, and clean files in `finally` blocks.
- The default shipped execution verifier is `python_compile`, matching the
  default `python:3.11-slim` image and command allowlist.
- `containers/verifier/Dockerfile` provides a dedicated non-root verifier image
  containing pytest, Ruff, and mypy for deployments that enable those checks.
- The verifier Dockerfile quotes version specifiers so shell metacharacters are
  never interpreted as redirections.
- Host execution remains an explicit unsafe opt-in. Container execution is the
  default runner.
- Container execution remains network-disabled, non-root, read-only, capability
  dropped, resource-limited, and digest pinning fails closed.
- Host/container subprocess wrappers propagate cancellation and terminate the
  underlying process/container rather than only cancelling the awaiting
  coroutine.

Primary files: `verification/execution_verifiers.py`, `execution/_bounded.py`,
`execution/runner.py`, `execution/container_runner.py`,
`containers/verifier/Dockerfile`.

## 6. Bounded execution worker pool

- Uses a real `asyncio.Queue(maxsize=queue_max)` so queue capacity checks and
  enqueue are atomic.
- Cancelling a caller prevents a still-queued job from later executing; active
  cancellation propagates to the runner task.
- Shutdown drains/cancels queued futures, bounds active-job drain time, and can
  signal every worker even when `queue_max < max_workers`.
- Parallelism tests use observed concurrent activity rather than brittle
  machine-dependent wall-clock thresholds.

Primary file: `execution/worker_pool.py`.

## 7. Persistent workflow/state-machine engine

The engine is intentionally described as a directed workflow graph/state
machine, not a DAG: the built-in repair workflow contains bounded cycles such
as `test -> reflect -> test`.

Statuses now distinguish the reason a run stopped:

- `PENDING`
- `RUNNING`
- `WAITING_APPROVAL` (`PAUSED` remains a compatibility alias)
- `INTERRUPTED`
- `COMPLETED`
- `FAILED`
- `CANCELLED`
- `REJECTED`

Safety semantics:

- At startup, stale `RUNNING` rows left by an ungraceful process exit are
  atomically converted to `INTERRUPTED`; any in-progress step attempt is also
  marked interrupted before retry.
- Automatic restart resumes only `INTERRUPTED` runs.
- `WAITING_APPROVAL` is never auto-approved after restart.
- `FAILED`, `CANCELLED`, and `REJECTED` work is never silently promoted or
  automatically retried.
- Approval futures are registered before `workflow.approval_requested` is
  published, eliminating the immediate-response lost-wakeup race.
- `cancel_run()` owns and cancels the actual in-flight asyncio task; cancellation
  cannot later be overwritten by a successful completion. Re-entrant/self-
  cancellation is tracked explicitly so a run cannot cancel itself and then
  fall through to `COMPLETED`.
- Service shutdown persists active work as `INTERRUPTED`, while approval waits
  remain `WAITING_APPROVAL`.
- Unknown graph edges fail explicitly rather than falling through to completion.
- Max-step exhaustion fails the run.
- Approval denial is terminal `REJECTED`.
- The built-in coding workflow fails closed: generation failure, unavailable
  execution, failed tests, or exhausted reflection can never become
  `COMPLETED`.

Primary files: `workflow/models.py`, `workflow/repository.py`,
`workflow/runner.py`, `workflow/builtins.py`.

## 8. Bootstrap, configuration, and packaging

- `event_bus.async_dispatch: true` now attaches the application's managed
  `TaskRegistry`; asynchronous dispatch changes runtime behavior instead of
  only toggling a flag.
- Reflection is wired to the instantiated `VerificationService`.
- Tool-registry policy is configurable under `tools`.
- `PyYAML`, `httpx`, and `jsonschema` are declared base dependencies.
- Optional extras declare Redis, tracing/export, semantic sqlite-vec, GUI, and
  development dependencies.
- Shipped configuration is offline-safe by default: the LLM gateway,
  conversation, feedback, reflection, workflow, and execution are disabled
  unless explicitly enabled.

## Regression coverage added by the hardening pass

Dedicated tests now cover:

- restart while waiting for approval;
- hard-crash recovery from stale `RUNNING` state;
- immediate checkpoint approval race;
- cancellation during an active workflow node and re-entrant self-cancellation;
- unknown workflow edges;
- async EventBus bootstrap wiring;
- embedding-space fingerprint isolation;
- invalid embedding-provider configuration;
- sqlite-vec native-table degradation path;
- detailed verifier failure propagation;
- concurrent verifier artifact isolation and cleanup;
- protocol-correct tool-call transcripts;
- malformed/type-invalid tool arguments;
- final-round tool side-effect suppression;
- Redis event identity/trace-context round trips and inbound echo suppression;
- OpenTelemetry current-span, real parent/child behavior, and async handler
  trace continuation;
- bounded worker shutdown with fewer queue slots than workers.

## Validation

The release validation record is maintained in `VALIDATION.txt`. The repaired
source is required to pass `compileall`, the complete pytest suite, and a
no-dependency wheel build before packaging.

Optional integration note: the release test environment has the OpenTelemetry
SDK/exporter available, so parent/child tracing is exercised against the real
SDK. Redis and sqlite-vec are optional and are not installed in this validation
environment; their protocol/degradation paths are covered by unit/regression
tests, but a live Redis server and native sqlite-vec extension are still
environmental deployment qualification items.
