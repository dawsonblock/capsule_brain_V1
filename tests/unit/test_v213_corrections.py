from pathlib import Path
import sys

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.models import ExecutionPolicy
from capsule_brain.execution.runner import ExecutionRunner
from capsule_brain.execution.service import ExecutionService
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.llm.tools import ToolCall, ToolSpec
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.builtins import build_plan_generate_test_reflect_workflow
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import WorkflowRunnerService


class FixedCodeProvider(LLMProvider):
    name = "fake"

    def __init__(self, code: str):
        self.code = code
        self.calls = 0

    async def generate(self, request, model_cfg):
        self.calls += 1
        if self.calls == 1:
            text = "Implement the requested behavior and satisfy acceptance tests."
        else:
            text = self.code
        return LLMResult(
            text=text,
            model="fake-model",
            provider="fake",
            latency_ms=1,
        )


def _gateway(provider: LLMProvider) -> LLMGateway:
    return LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake-model",
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
        providers={"fake": provider},
    )


@pytest.mark.asyncio
async def test_build_application_rejects_direct_host_execution_bypass(tmp_path: Path):
    """Direct dict construction must enforce the same host safety gate as YAML."""
    with pytest.raises(ValueError, match="unsafe_allow_host_execution"):
        build_application(
            {
                "execution": {
                    "enable": True,
                    "allow": True,
                    "runner": "host",
                    "cwd_root": str(tmp_path / "sandbox"),
                }
            }
        )


def test_smoke_workflow_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="allow_smoke_success"):
        build_plan_generate_test_reflect_workflow(
            services=None,
            verification_mode="smoke",
        )


@pytest.mark.asyncio
async def test_default_workflow_does_not_equate_exit_zero_with_success(tmp_path: Path):
    """A syntactically valid no-op must fail without an acceptance contract."""
    bus = LocalEventBus()
    await bus.start()
    root = tmp_path / "sandbox"
    root.mkdir()
    executable = Path(sys.executable).name
    execution = ExecutionService(
        bus,
        cfg={"allow": True, "cwd_root": str(root), "allowed_commands": [executable]},
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
    gateway = _gateway(FixedCodeProvider("pass"))
    await gateway.start()

    runner = WorkflowRunnerService(
        bus,
        repository=WorkflowRepository(str(tmp_path / "wf.sqlite")),
    )
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services={"llm_gateway": gateway, "execution": execution},
            max_iterations=1,
        )
    )
    await runner.start()
    try:
        run = await runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="return the mathematically correct result",
        )
        assert run.status == WorkflowStatus.FAILED
        assert run.state.extra["test_passed"] is False
        assert run.state.extra["verification_kind"] == "acceptance"
        assert run.state.extra["verification_contract_present"] is False
        assert "acceptance verification missing" in run.state.stderr
    finally:
        await runner.stop()
        await gateway.stop()
        await execution.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_acceptance_contract_can_promote_verified_solution(tmp_path: Path):
    bus = LocalEventBus()
    await bus.start()
    root = tmp_path / "sandbox"
    root.mkdir()
    executable = Path(sys.executable).name
    execution = ExecutionService(
        bus,
        cfg={"allow": True, "cwd_root": str(root), "allowed_commands": [executable]},
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
    gateway = _gateway(FixedCodeProvider("def add(a, b):\n    return a + b\n"))
    await gateway.start()

    runner = WorkflowRunnerService(
        bus,
        repository=WorkflowRepository(str(tmp_path / "wf.sqlite")),
    )
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services={"llm_gateway": gateway, "execution": execution}
        )
    )
    await runner.start()
    try:
        run = await runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="implement add(a, b)",
            metadata={
                "verification": {
                    "files": {
                        "acceptance.py": (
                            "from solution import add\n"
                            "assert add(2, 3) == 5\n"
                            "assert add(-2, 2) == 0\n"
                        )
                    },
                    "command": [executable, "acceptance.py"],
                }
            },
        )
        assert run.status == WorkflowStatus.COMPLETED
        assert run.state.extra["test_passed"] is True
        assert run.state.extra["verification_kind"] == "acceptance"
        assert run.state.extra["verification_contract_present"] is True
        assert run.state.extra["exit_code"] == 0
    finally:
        await runner.stop()
        await gateway.stop()
        await execution.stop()
        await bus.stop()


class ConversationToolProvider(LLMProvider):
    name = "fake"

    async def generate(self, request, model_cfg):
        if request.tools and not request.tool_results:
            return LLMResult(
                text="",
                model="fake-model",
                provider="fake",
                latency_ms=1,
                tool_calls=[
                    ToolCall(
                        id="calc_1",
                        name="double",
                        arguments={"value": 21},
                        raw_arguments='{"value":21}',
                    )
                ],
                finish_reason="tool_calls",
            )
        value = request.tool_results[-1]["content"] if request.tool_results else "none"
        return LLMResult(
            text=f"tool result: {value}",
            model="fake-model",
            provider="fake",
            latency_ms=1,
        )


@pytest.mark.asyncio
async def test_conversation_service_can_use_registered_tools(tmp_path: Path):
    app = build_application(
        {
            "memory": {"db_path": str(tmp_path / "memory.sqlite")},
            "memory_consolidator": {"enable": False},
            "learning": {"db_path": str(tmp_path / "experience.sqlite")},
            "llm_gateway": {
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {
                        "provider": "fake",
                        "model_name": "fake-model",
                        "capabilities": ["text", "tools"],
                    }
                },
                "routing": {"routes": {"chat": {"models": ["fake"]}}},
            },
            "conversation": {
                "enable": True,
                "use_tools": True,
                "db_path": str(tmp_path / "conversation.sqlite"),
            },
            "autolearn": {"enable": False},
            "feedback": {"enable": False},
            "reflection": {"enable": False},
            "verification": {"enable": False},
            "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
        },
        llm_providers={"fake": ConversationToolProvider()},
    )
    registry = app.services.require("tool_registry")

    async def double(args):
        return {"value": args["value"] * 2}

    registry.register(
        ToolSpec(
            name="double",
            description="Double an integer.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        double,
    )

    await app.start()
    try:
        conversation = app.services.require("conversation")
        response = await conversation.respond(
            conversation_id="tool-conversation",
            text="double 21",
        )
        assert "42" in response.text
        health = await conversation.health()
        assert health.details["use_tools"] is True
        assert health.details["registered_tools"] == ["double"]
    finally:
        await app.stop()


@pytest.mark.asyncio
async def test_failing_acceptance_script_blocks_completion_despite_clean_solution(
    tmp_path: Path,
):
    """Acceptance evidence, not solution.py's clean execution, gates promotion."""
    bus = LocalEventBus()
    await bus.start()
    root = tmp_path / "sandbox"
    root.mkdir()
    executable = Path(sys.executable).name
    execution = ExecutionService(
        bus,
        cfg={"allow": True, "cwd_root": str(root), "allowed_commands": [executable]},
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
    gateway = _gateway(FixedCodeProvider("def add(a, b):\n    return a + b\n"))
    await gateway.start()

    runner = WorkflowRunnerService(
        bus,
        repository=WorkflowRepository(str(tmp_path / "wf.sqlite")),
    )
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services={"llm_gateway": gateway, "execution": execution},
            max_iterations=1,
        )
    )
    await runner.start()
    try:
        run = await runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="implement add(a, b)",
            metadata={
                "verification": {
                    "files": {
                        "acceptance.py": (
                            "from solution import add\n"
                            "assert add(2, 3) == 999\n"
                        )
                    },
                    "command": [executable, "acceptance.py"],
                }
            },
        )
        assert run.status == WorkflowStatus.FAILED
        assert run.state.extra["test_passed"] is False
        assert run.state.extra["verification_kind"] == "acceptance"
        assert run.state.extra["verification_contract_present"] is True
        assert run.state.extra["exit_code"] != 0
        assert "AssertionError" in run.state.stderr
    finally:
        await runner.stop()
        await gateway.stop()
        await execution.stop()
        await bus.stop()
