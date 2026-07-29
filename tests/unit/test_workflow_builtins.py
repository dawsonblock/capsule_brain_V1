import sys
import pytest
from pathlib import Path

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.models import ExecutionPolicy
from capsule_brain.execution.runner import ExecutionRunner
from capsule_brain.execution.service import ExecutionService
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository
from capsule_brain.workflow.builtins import (
    build_plan_generate_test_reflect_workflow,
)
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import WorkflowRunnerService


# ---------------------------------------------------------------------------
# Offline mode (no LLM, no execution)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_workflow_offline_mode_fails_closed(tmp_path):
    """No LLM/executor must never be promoted as a successful workflow."""
    bus = LocalEventBus()
    await bus.start()
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(services=None)
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.FAILED
    assert "[offline]" in run.state.plan
    assert run.state.code == ""
    assert run.state.extra.get("generation_failed") is True
    assert run.state.extra.get("test_passed") is False
    assert run.state.iterations == 3
    assert run.stop_reason == "node_error:failed"

    await runner.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# With execution service (test passes)
# ---------------------------------------------------------------------------

class StubLLMProvider(LLMProvider):
    """Returns canned code for plan and generate nodes."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        # The first call is the plan node; subsequent calls are generate.
        if self.calls == 1:
            return LLMResult(
                text="1. Write a print statement",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        return LLMResult(
            text="print('hello from workflow')",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_default_workflow_with_execution_passes(tmp_path):
    """The full pipeline: LLM generates code, execution runs it, test passes."""
    bus = LocalEventBus()
    await bus.start()

    # Set up execution service with a real runner.
    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    # Set up LLM gateway with a stub provider.
    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": StubLLMProvider()},
    )
    await gateway.start()

    services = {
        "llm_gateway": gateway,
        "execution": execution,
    }
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services,
            verification_mode="smoke",
            allow_smoke_success=True,
        )
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.COMPLETED
    assert run.state.plan == "1. Write a print statement"
    assert "print('hello from workflow')" in run.state.code
    assert "hello from workflow" in run.state.stdout
    assert run.state.extra.get("test_passed") is True
    assert run.state.iterations == 0  # no reflect loop needed

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# With execution service (test fails -> reflect loop)
# ---------------------------------------------------------------------------

class FailingCodeProvider(LLMProvider):
    """Returns broken code on the first generate, fixed code after reflect."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.calls == 1:
            # Plan
            return LLMResult(
                text="1. Write a print statement",
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        if self.calls == 2:
            # Generate: broken code (syntax error)
            return LLMResult(
                text="print('hello'",  # missing closing paren
                model="fake",
                provider="fake",
                latency_ms=1,
            )
        # Reflect (call 3): fixed code
        return LLMResult(
            text="print('hello')",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_default_workflow_reflect_loop_fixes_failure(tmp_path):
    """Test fails -> reflect -> test passes on the second iteration."""
    bus = LocalEventBus()
    await bus.start()

    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": FailingCodeProvider()},
    )
    await gateway.start()

    services = {"llm_gateway": gateway, "execution": execution}
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services,
            max_iterations=3,
            verification_mode="smoke",
            allow_smoke_success=True,
        )
    )
    await runner.start()

    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="write a hello world script",
    )

    assert run.status == WorkflowStatus.COMPLETED
    # The reflect loop should have run once.
    assert run.state.iterations == 1
    # The final code should be the fixed version.
    assert "print('hello')" in run.state.code
    # The test should have passed on the second attempt.
    assert run.state.extra.get("test_passed") is True
    assert "hello" in run.state.stdout
    # Steps: plan, generate, test (fail), reflect, test (pass), complete
    node_sequence = [s.node_name for s in run.steps]
    assert node_sequence == ["plan", "generate", "test", "reflect", "test", "complete"]

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_workflow_with_approval_checkpoint(tmp_path):
    """require_approval=True inserts a checkpoint after test passes."""
    bus = LocalEventBus()
    await bus.start()

    root = tmp_path / "sandbox"
    root.mkdir()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [Path(sys.executable).name],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(Path(sys.executable).name,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    gateway = LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": StubLLMProvider()},
    )
    await gateway.start()

    services = {"llm_gateway": gateway, "execution": execution}
    repo = WorkflowRepository(str(tmp_path / "wf.sqlite"))
    runner = WorkflowRunnerService(bus, repository=repo)
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services,
            require_approval=True,
            verification_mode="smoke",
            allow_smoke_success=True,
        )
    )
    await runner.start()

    import asyncio
    approval_events = []
    approval_ready = asyncio.Event()

    def on_approval(e):
        approval_events.append(e.payload)
        approval_ready.set()

    bus.subscribe("workflow.approval_requested", on_approval)

    task = asyncio.create_task(
        runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="write a hello world script",
        )
    )
    await asyncio.wait_for(approval_ready.wait(), timeout=5.0)

    # The run should be parked at the approve checkpoint.
    assert len(approval_events) == 1
    assert approval_events[0]["node"] == "approve"
    run_id = approval_events[0]["run_id"]

    paused = await runner.repository.get_run(run_id)
    assert paused.status == WorkflowStatus.PAUSED

    # Approve and let it complete.
    await runner.resume_run(run_id, approved=True)
    run = await asyncio.wait_for(task, timeout=5.0)
    assert run.status == WorkflowStatus.COMPLETED

    # Steps: plan, generate, test, approve, complete
    node_sequence = [s.node_name for s in run.steps]
    assert node_sequence == ["plan", "generate", "test", "approve", "complete"]

    await runner.stop()
    await gateway.stop()
    await execution.stop()
    await bus.stop()


# ---------------------------------------------------------------------------
# Bootstrap integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bootstrap_registers_workflow_service(tmp_path):
    """build_application with workflow.enable=true registers the runner."""
    from capsule_brain.runtime.bootstrap import build_application

    cfg = {
        "workflow": {"enable": True, "db_path": str(tmp_path / "wf.sqlite")},
    }
    app = build_application(cfg)
    assert "workflow_runner" in app.services.names()
    runner = app.services.get("workflow_runner")
    assert runner is not None
    assert runner.get_workflow("plan_generate_test_reflect") is not None


# ---------------------------------------------------------------------------
# v2.13.1 verification-coverage completion
# ---------------------------------------------------------------------------

class _AlwaysCodeProvider(LLMProvider):
    """Deterministic provider for acceptance-boundary regression tests."""

    name = "fake"

    async def generate(self, request, model_cfg):
        return LLMResult(
            text="def add(a, b):\n    return a + b\n",
            model="fake",
            provider="fake",
            latency_ms=1,
        )


def _acceptance_gateway() -> LLMGateway:
    return LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": _AlwaysCodeProvider()},
    )


async def _run_acceptance_case(
    tmp_path: Path,
    *,
    metadata: dict,
    execution,
    verification=None,
):
    bus = LocalEventBus()
    await bus.start()
    gateway = _acceptance_gateway()
    await gateway.start()
    runner = WorkflowRunnerService(
        bus,
        repository=WorkflowRepository(str(tmp_path / "wf.sqlite")),
    )
    services = {"llm_gateway": gateway, "execution": execution}
    if verification is not None:
        services["verification"] = verification
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services=services,
            max_iterations=1,
        )
    )
    await runner.start()
    try:
        return await runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="implement add(a, b)",
            metadata=metadata,
        )
    finally:
        await runner.stop()
        await gateway.stop()
        await bus.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("path_case", ["parent", "absolute", "solution_collision"])
async def test_acceptance_verification_rejects_path_traversal(tmp_path, path_case):
    """Verification assets cannot escape or overwrite the generated artifact."""
    bus = LocalEventBus()
    await bus.start()
    root = tmp_path / "sandbox"
    root.mkdir()
    executable = Path(sys.executable).name
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [executable],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(executable,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    if path_case == "parent":
        bad_name = "../evil.py"
    elif path_case == "absolute":
        bad_name = str((tmp_path / "outside.py").resolve())
    else:
        bad_name = "solution.py"

    # _run_acceptance_case owns a different bus for the workflow, but the
    # execution service itself does not require the same bus to execute.
    try:
        run = await _run_acceptance_case(
            tmp_path,
            execution=execution,
            metadata={
                "verification": {
                    "files": {bad_name: "print('escaped')"},
                    "command": [executable, "acceptance.py"],
                }
            },
        )
        assert run.status == WorkflowStatus.FAILED
        assert run.state.extra["test_passed"] is False
        assert run.state.extra["verification_kind"] == "acceptance"
        assert run.state.extra["verification_contract_present"] is True
        assert not (root / "evil.py").exists()
        assert not (tmp_path / "outside.py").exists()
        assert not (root / f"workflow_{run.state.run_id}").exists()
    finally:
        await execution.stop()
        await bus.stop()


class _FailingExecutionService:
    def __init__(self, root: Path):
        self.is_running = True
        self.policy = ExecutionPolicy(
            allow=True,
            cwd_root=str(root),
            allowed_commands=(Path(sys.executable).name,),
            timeout_s=5,
        )

    async def execute(self, request):
        raise RuntimeError("daemon unreachable")


class _ExplodingVerificationService:
    is_running = True

    async def verify(self, **kwargs):
        raise RuntimeError("verifier backend unavailable")


@pytest.mark.asyncio
async def test_acceptance_verification_fails_closed_on_execution_infrastructure_error(
    tmp_path,
):
    """Execution and VerificationService infrastructure errors both fail closed."""
    executable = Path(sys.executable).name

    # Variant 1: the acceptance execution backend itself raises instead of
    # returning a normal non-zero ExecutionResult.
    execution_case = tmp_path / "execution_failure"
    execution_case.mkdir()
    execution_root = execution_case / "sandbox"
    execution_root.mkdir()
    run = await _run_acceptance_case(
        execution_case,
        execution=_FailingExecutionService(execution_root),
        metadata={
            "verification": {
                "files": {
                    "acceptance.py": (
                        "from solution import add\n"
                        "assert add(2, 3) == 5\n"
                    )
                },
                "command": [executable, "acceptance.py"],
            }
        },
    )
    assert run.status == WorkflowStatus.FAILED
    assert run.state.extra["test_passed"] is False
    assert run.state.extra["verification_kind"] == "acceptance"
    assert "daemon unreachable" in run.state.stderr

    # Variant 2: the shared VerificationService raises before acceptance
    # execution. A real executor is present to isolate this failure stage.
    verification_case = tmp_path / "verification_failure"
    verification_case.mkdir()
    verification_root = verification_case / "sandbox"
    verification_root.mkdir()
    bus = LocalEventBus()
    await bus.start()
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(verification_root),
            "allowed_commands": [executable],
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(verification_root),
                allowed_commands=(executable,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()
    try:
        run = await _run_acceptance_case(
            verification_case,
            execution=execution,
            verification=_ExplodingVerificationService(),
            metadata={
                "verification": {
                    "files": {
                        "acceptance.py": (
                            "from solution import add\n"
                            "assert add(2, 3) == 5\n"
                        )
                    },
                    "command": [executable, "acceptance.py"],
                }
            },
        )
        assert run.status == WorkflowStatus.FAILED
        assert run.state.extra["test_passed"] is False
        assert run.state.extra["verification_kind"] == "static"
        assert "verifier backend unavailable" in run.state.stderr
    finally:
        await execution.stop()
        await bus.stop()
