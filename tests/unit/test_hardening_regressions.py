from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.models import MemoryRecord
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.verification.execution_verifiers import PythonCompileVerifier
from capsule_brain.verification.models import (
    VerificationCheck,
    VerificationStatus,
)
from capsule_brain.verification.repository import VerificationRepository
from capsule_brain.verification.service import VerificationService
from capsule_brain.verification.verifiers import Verifier
from capsule_brain.workflow.models import WorkflowState, WorkflowStatus
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import Workflow, WorkflowNode, WorkflowRunnerService


async def _make_workflow_runner(tmp_path: Path, *, cfg: dict | None = None):
    bus = LocalEventBus()
    await bus.start()
    repo = WorkflowRepository(str(tmp_path / "workflows.sqlite"))
    runner = WorkflowRunnerService(bus, cfg=cfg or {}, repository=repo)
    return bus, runner


def _approval_workflow(name: str, marker: list[str] | None = None) -> Workflow:
    marker = marker if marker is not None else []

    async def before(state: WorkflowState) -> WorkflowState:
        marker.append("before")
        state.errors.append("before")
        return state

    async def after(state: WorkflowState) -> WorkflowState:
        marker.append("after")
        state.errors.append("after")
        return state

    return Workflow(
        name=name,
        nodes=[
            WorkflowNode("before", before, next_node="after", checkpoint=lambda _: True),
            WorkflowNode("after", after, next_node=None),
        ],
        entry="before",
    )


@pytest.mark.asyncio
async def test_immediate_checkpoint_approval_cannot_be_lost(tmp_path):
    """The approval future must exist before approval_requested is published."""
    bus, runner = await _make_workflow_runner(tmp_path)
    runner.register_workflow(_approval_workflow("immediate"))

    async def approve_immediately(event):
        await runner.resume_run(event.payload["run_id"], approved=True)

    bus.subscribe("workflow.approval_requested", approve_immediately)
    await runner.start()
    try:
        run = await asyncio.wait_for(
            runner.start_run(workflow_name="immediate", goal="test"),
            timeout=2.0,
        )
        assert run.status == WorkflowStatus.COMPLETED
        assert run.state.approved is True
        assert run.state.errors == ["before", "after"]
    finally:
        await runner.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_restart_never_auto_approves_waiting_checkpoint(tmp_path):
    db_path = tmp_path / "workflows.sqlite"
    markers: list[str] = []

    bus1 = LocalEventBus()
    await bus1.start()
    runner1 = WorkflowRunnerService(
        bus1,
        cfg={"auto_resume": True},
        repository=WorkflowRepository(str(db_path)),
    )
    runner1.register_workflow(_approval_workflow("restart-safe", markers))
    approval_seen = asyncio.Event()
    run_id: str | None = None

    def record_approval(event):
        nonlocal run_id
        run_id = event.payload["run_id"]
        approval_seen.set()

    bus1.subscribe("workflow.approval_requested", record_approval)
    await runner1.start()
    task = asyncio.create_task(
        runner1.start_run(workflow_name="restart-safe", goal="test")
    )
    await asyncio.wait_for(approval_seen.wait(), timeout=2.0)
    assert run_id is not None
    waiting = await runner1.repository.get_run(run_id)
    assert waiting is not None
    assert waiting.status == WorkflowStatus.WAITING_APPROVAL
    assert markers == ["before"]

    # Simulate a clean process shutdown while the human gate is pending.
    await runner1.stop()
    await asyncio.wait_for(task, timeout=2.0)
    await bus1.stop()

    # A new process may auto-resume interrupted work, but not approval waits.
    bus2 = LocalEventBus()
    await bus2.start()
    runner2 = WorkflowRunnerService(
        bus2,
        cfg={"auto_resume": True},
        repository=WorkflowRepository(str(db_path)),
    )
    runner2.register_workflow(_approval_workflow("restart-safe", markers))
    await runner2.start()
    try:
        await asyncio.sleep(0)
        persisted = await runner2.repository.get_run(run_id)
        assert persisted is not None
        assert persisted.status == WorkflowStatus.WAITING_APPROVAL
        assert markers == ["before"]
        assert await runner2.repository.list_pending_resume() == []

        # Only an explicit operator decision may cross the checkpoint.
        resumed = await runner2.resume_run(run_id, approved=True)
        assert resumed is not None
        assert resumed.status == WorkflowStatus.COMPLETED
        assert markers == ["before", "after"]
    finally:
        await runner2.stop()
        await bus2.stop()


@pytest.mark.asyncio
async def test_cancel_active_workflow_stops_task_and_cannot_be_overwritten(tmp_path):
    bus, runner = await _make_workflow_runner(tmp_path)
    entered = asyncio.Event()
    reached_after = False

    async def slow(state: WorkflowState) -> WorkflowState:
        entered.set()
        await asyncio.sleep(60)
        return state

    async def after(state: WorkflowState) -> WorkflowState:
        nonlocal reached_after
        reached_after = True
        return state

    runner.register_workflow(
        Workflow(
            name="active-cancel",
            nodes=[
                WorkflowNode("slow", slow, next_node="after"),
                WorkflowNode("after", after, next_node=None),
            ],
        )
    )
    await runner.start()
    try:
        task = asyncio.create_task(
            runner.start_run(workflow_name="active-cancel", goal="test")
        )
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        running = await runner.repository.list_runs(status=WorkflowStatus.RUNNING)
        assert len(running) == 1
        run_id = running[0].id

        cancelled = await runner.cancel_run(run_id)
        assert cancelled is not None
        assert cancelled.status == WorkflowStatus.CANCELLED

        returned = await asyncio.wait_for(task, timeout=2.0)
        assert returned.status == WorkflowStatus.CANCELLED
        assert reached_after is False

        persisted = await runner.repository.get_run(run_id)
        assert persisted is not None
        assert persisted.status == WorkflowStatus.CANCELLED
        assert persisted.stop_reason == "cancelled"
        assert persisted.steps[-1].status == "cancelled"
    finally:
        await runner.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_unknown_workflow_edge_fails_instead_of_completing(tmp_path):
    bus, runner = await _make_workflow_runner(tmp_path)

    async def node(state: WorkflowState) -> WorkflowState:
        return state

    runner.register_workflow(
        Workflow(
            name="bad-edge",
            nodes=[WorkflowNode("start", node, next_node="does_not_exist")],
        )
    )
    await runner.start()
    try:
        run = await runner.start_run(workflow_name="bad-edge", goal="test")
        assert run.status == WorkflowStatus.FAILED
        assert run.stop_reason == "unknown_node:does_not_exist"
    finally:
        await runner.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_bootstrap_async_dispatch_is_actually_managed(tmp_path):
    app = build_application(
        {
            "event_bus": {"async_dispatch": True},
            "memory": {
                "db_path": str(tmp_path / "memory.sqlite"),
                "auto_index_embeddings": False,
            },
            "memory_consolidator": {"enable": False},
            "learning": {"db_path": str(tmp_path / "experience.sqlite")},
            "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
            "verification": {
                "enable": True,
                "db_path": str(tmp_path / "verification.sqlite"),
            },
        }
    )
    bus = app.services.require("event_bus")
    assert bus._task_registry is app.tasks

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_handler(_event):
        entered.set()
        await release.wait()

    bus.subscribe("hardening.block", blocking_handler)
    await app.start()
    try:
        # If bootstrap failed to attach TaskRegistry, publish would block here.
        from capsule_brain.events.models import EventEnvelope

        await asyncio.wait_for(
            bus.publish(EventEnvelope(event_type="hardening.block", source="test", payload={})),
            timeout=0.5,
        )
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        release.set()
        await asyncio.sleep(0)
    finally:
        release.set()
        await app.stop()


@pytest.mark.asyncio
async def test_embedding_fingerprints_isolate_vector_spaces(tmp_path):
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    await repo.start()
    try:
        first = MemoryRecord(text="first", source="test")
        second = MemoryRecord(text="second", source="test")
        await repo.create(first)
        await repo.create(second)
        await repo.index_embedding(first.id, [1.0, 0.0], fingerprint="space:A")
        await repo.index_embedding(second.id, [1.0, 0.0], fingerprint="space:B")

        a = await repo.search_semantic([1.0, 0.0], limit=8, fingerprint="space:A")
        b = await repo.search_semantic([1.0, 0.0], limit=8, fingerprint="space:B")
        assert [item.id for item in a] == [first.id]
        assert [item.id for item in b] == [second.id]
    finally:
        await repo.stop()


def test_unknown_embedding_provider_fails_loudly(tmp_path):
    bus = LocalEventBus()
    with pytest.raises(ValueError, match="Unknown memory embedding provider"):
        MemoryService(
            bus,
            cfg={
                "db_path": str(tmp_path / "memory.sqlite"),
                "embedding": {"provider": "typo-provider"},
            },
        )


@pytest.mark.asyncio
async def test_sqlite_vec_table_creation_falls_back_without_name_errors(tmp_path):
    """Exercise the native-table branch even when sqlite-vec is not installed."""
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    await repo.start()
    try:
        repo._vec_available = True
        # Plain SQLite has no vec0 module. This should degrade cleanly, not
        # reference search-only variables from table creation.
        repo._ensure_vec_table(2)
        assert repo._vec_available is False
    finally:
        await repo.stop()


class _DetailedFailureVerifier(Verifier):
    name = "detailed_failure"

    async def verify(self, subject: str, metadata: dict) -> VerificationCheck:
        return VerificationCheck(
            verifier=self.name,
            status=VerificationStatus.FAIL,
            summary="assertion failed",
            details={"stdout": "actual=1", "stderr": "expected=2", "diff": "1 != 2"},
        )


@pytest.mark.asyncio
async def test_verification_failure_event_preserves_check_details(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    service = VerificationService(
        bus,
        repository=VerificationRepository(str(tmp_path / "verification.sqlite")),
        verifiers=[_DetailedFailureVerifier()],
    )
    failed_events: list[dict] = []
    bus.subscribe("verification.failed", lambda event: failed_events.append(event.payload))
    await service.start()
    try:
        result = await service.verify(subject="x", source="test")
        assert result.status == VerificationStatus.FAIL
        assert len(failed_events) == 1
        payload = failed_events[0]
        assert payload["check_details"]["verifier"] == "detailed_failure"
        assert payload["check_details"]["details"]["stderr"] == "expected=2"
        assert payload["failed_checks"][0]["details"]["diff"] == "1 != 2"
    finally:
        await service.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_execution_verifier_uses_unique_relative_artifacts(tmp_path):
    from capsule_brain.execution.models import ExecutionPolicy, ExecutionResult, now

    class FakeExecution:
        def __init__(self):
            self.policy = ExecutionPolicy(allow=True, cwd_root=str(tmp_path / "sandbox"))
            self.artifacts: list[str] = []
            self.both_seen = asyncio.Event()

        async def execute(self, request):
            artifact = request.command[-1]
            self.artifacts.append(artifact)
            assert not Path(artifact).is_absolute()
            assert (Path(request.cwd) / artifact).exists()
            if len(self.artifacts) >= 2:
                self.both_seen.set()
            await asyncio.wait_for(self.both_seen.wait(), timeout=1.0)
            return ExecutionResult(
                request_id=request.id,
                command=request.command,
                cwd=request.cwd or "",
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
                duration_ms=1.0,
                started_at=now(),
                completed_at=now(),
            )

    execution = FakeExecution()
    verifier = PythonCompileVerifier(execution)
    checks = await asyncio.gather(
        verifier.verify("x = 1", {"content_type": "python"}),
        verifier.verify("y = 2", {"content_type": "python"}),
    )
    assert all(check.status == VerificationStatus.PASS for check in checks)
    assert len(set(execution.artifacts)) == 2
    assert not any((Path(execution.policy.cwd_root) / name).exists() for name in execution.artifacts)


def test_redis_bridge_round_trip_preserves_event_identity_and_trace_context():
    from capsule_brain.events.models import EventEnvelope
    from capsule_brain.events.redis_bridge import RedisBridge
    from uuid import uuid4

    bus = LocalEventBus()
    bridge = RedisBridge(bus)
    correlation_id = uuid4()
    original = EventEnvelope(
        event_type="trace.test",
        source="unit",
        correlation_id=correlation_id,
        trace_context={
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor=value",
        },
        payload={"answer": 42},
    )

    encoded = bridge._encode_event(original)
    decoded = bridge._decode_event("trace.test", encoded)

    assert decoded.event_id == original.event_id
    assert decoded.correlation_id == correlation_id
    assert decoded.created_at == original.created_at
    assert decoded.event_type == original.event_type
    assert decoded.source == original.source
    assert decoded.payload == original.payload
    assert decoded.trace_context == original.trace_context


@pytest.mark.asyncio
async def test_hard_crash_running_run_is_marked_interrupted_then_resumed(tmp_path):
    from capsule_brain.workflow.models import WorkflowRun, WorkflowStepRecord

    db_path = str(tmp_path / "crash-recovery.sqlite")
    repo = WorkflowRepository(db_path)
    await repo.start()
    state = WorkflowState(goal="recover", current_node="work")
    stale = WorkflowRun(
        id=state.run_id,
        workflow_name="crash-recovery",
        status=WorkflowStatus.RUNNING,
        state=state,
        current_node="work",
    )
    await repo.save_run(stale)
    await repo.save_step(
        WorkflowStepRecord(
            run_id=stale.id,
            node_name="work",
            index=0,
            started_at="2026-07-26T00:00:00+00:00",
        )
    )
    # Simulate the durable state a hard process death leaves behind.
    await repo.stop()

    bus = LocalEventBus()
    await bus.start()
    resumed = asyncio.Event()

    async def work(state: WorkflowState) -> WorkflowState:
        state.errors.append("recovered")
        return state

    runner = WorkflowRunnerService(
        bus,
        cfg={"db_path": db_path, "auto_resume": True},
        repository=WorkflowRepository(db_path),
    )
    runner.register_workflow(
        Workflow(name="crash-recovery", nodes=[WorkflowNode("work", work)])
    )

    async def on_complete(event):
        if event.payload.get("run_id") == stale.id:
            resumed.set()

    bus.subscribe("workflow.completed", on_complete)
    await runner.start()
    try:
        await asyncio.wait_for(resumed.wait(), timeout=2.0)
        recovered = await runner.repository.get_run(stale.id)
        assert recovered is not None
        assert recovered.status == WorkflowStatus.COMPLETED
        assert recovered.state.errors == ["recovered"]
        assert len(recovered.steps) == 2
        assert recovered.steps[0].status == "interrupted"
        assert recovered.steps[0].error == "process_restart"
        assert recovered.steps[1].status == "passed"
    finally:
        await runner.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_workflow_can_cancel_itself_without_overwriting_terminal_state(tmp_path):
    bus, runner = await _make_workflow_runner(tmp_path)

    async def self_cancel(state: WorkflowState) -> WorkflowState:
        await bus.publish(
            EventEnvelope(
                event_type="workflow.cancel",
                source="workflow-node",
                payload={"run_id": state.run_id},
            )
        )
        return state

    from capsule_brain.events.models import EventEnvelope
    runner.register_workflow(
        Workflow(name="self-cancel", nodes=[WorkflowNode("cancel", self_cancel)])
    )
    await runner.start()
    try:
        run = await runner.start_run(workflow_name="self-cancel", goal="stop")
        assert run.status == WorkflowStatus.CANCELLED
        persisted = await runner.repository.get_run(run.id)
        assert persisted is not None
        assert persisted.status == WorkflowStatus.CANCELLED
        assert persisted.stop_reason == "cancelled"
        assert persisted.steps[-1].status == "cancelled"
    finally:
        await runner.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_redis_bridge_does_not_echo_inbound_event_back_out():
    from capsule_brain.events.models import EventEnvelope
    from capsule_brain.events.redis_bridge import RedisBridge

    class FakeRedis:
        def __init__(self):
            self.published = []

        async def publish(self, channel, payload):
            self.published.append((channel, payload))

    bus = LocalEventBus()
    bridge = RedisBridge(bus, cfg={"outbound_topics": ["same"]})
    bridge._redis = FakeRedis()
    event = EventEnvelope(event_type="same", source="remote", payload={"x": 1})
    bridge._remember_inbound(event.event_id)

    handler = bridge._make_outbound_handler("same")
    await handler(event)
    assert bridge._redis.published == []

    local_event = EventEnvelope(event_type="same", source="local", payload={"x": 2})
    await handler(local_event)
    assert len(bridge._redis.published) == 1
