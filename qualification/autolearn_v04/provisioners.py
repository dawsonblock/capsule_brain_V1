"""Real per-action isolated service factory + task provisioners for v0.4.0.

Section 9.2: every task/action execution receives its own isolated service
bundle (fresh MemoryService, ExperienceStore, ToolRegistry, WorkflowRunner,
ConversationService, ExecutiveController, isolated DBs + filesystem).

Required isolation:
  * separate temporary directory;
  * separate in-memory state;
  * separate event-bus namespace;
  * separate database transaction or database copy;
  * separate tool sandbox;
  * deterministic task seed;
  * no cross-action cache contamination;
  * no reuse of generated answer artifacts;
  * no shared memory mutation.

No global MemoryService/ToolRegistry/WorkflowRunner/DB is reused across
counterfactual actions.
"""
from __future__ import annotations

import secrets as _secrets
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.controller import (
    ActionDispatcher,
    DispatcherResult,
    ExecutiveController,
    MemoryExperienceSink,
)
from capsule_brain.autolearn.schema import Action, ExecutiveState
from capsule_brain.conversation.models import RequestContext
from capsule_brain.conversation.service import ConversationService, _ConversationActionDispatcher
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.service import ExecutionService
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.tools import ToolRegistry, ToolSpec
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.registry import ServiceRegistry
from capsule_brain.runtime.service import ServiceState
from capsule_brain.workflow.builtins import build_plan_generate_test_reflect_workflow
from capsule_brain.workflow.runner import WorkflowRunnerService

from .config import PROVIDER_INFRASTRUCTURE, PROVIDER_LOCAL_TRANSFORMERS, QualificationConfig
from .grounded_provider import GroundedQualificationProvider


@dataclass(slots=True)
class IsolatedEnvironment:
    env_id: str
    root: Path
    memory_db: Path
    experience_db: Path
    workflow_db: Path
    conversation_db: Path
    fs_root: Path
    exec_dir: Path
    seed: int
    _cleaned: bool = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        shutil.rmtree(self.root, ignore_errors=True)
        self._cleaned = True


@dataclass(slots=True)
class ProvisionedTask:
    task_id: str
    family: str
    verifier_state: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    setup_for_runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QualificationServices:
    """A fresh, isolated bundle of real Capsule Brain services for one action."""

    environment: IsolatedEnvironment
    bus: LocalEventBus
    memory: MemoryService
    tool_registry: ToolRegistry
    workflow_runner: WorkflowRunnerService
    execution: ExecutionService
    conversation: ConversationService
    controller: ExecutiveController
    registry: ServiceRegistry
    provisioned: ProvisionedTask
    provider: Any

    async def execute_action(
        self,
        *,
        action: Action,
        state: ExecutiveState,
        task: dict[str, Any],
    ) -> DispatcherResult:
        conv_id = f"cf_{task['task_id']}_{action.value}_{uuid.uuid4().hex[:8]}"
        md = self._model_visible_metadata(task)
        ctx = RequestContext(
            request_id=conv_id,
            conversation_id=conv_id,
            prompt=task["prompt"],
            history=(),
            memory_records=(),
            available_tools=tuple(state.available_tools),
            user_turn_id="cf_turn",
            metadata=dict(md),
        )
        return await self.conversation.execute_executive_action(
            action=action,
            state=state,
            conversation_id=conv_id,
            request_context=ctx,
        )

    def _model_visible_metadata(self, task: dict[str, Any]) -> dict[str, Any]:
        md = {
            "text": task["prompt"],
            "task_id": task["task_id"],
            "task_family": task["family"],
        }
        setup = dict(task.get("setup_spec", {}) or {})
        for key in ("secret", "expected_tool_output", "expected_token", "nonce",
                    "solution_code", "conflict_secret", "stale_secret"):
            setup.pop(key, None)
        if task["family"] == "workflow_required":
            acceptance_code = (task.get("setup_spec", {}) or {}).get("acceptance_code", "")
            if acceptance_code:
                md["acceptance_code"] = acceptance_code
        md.update(setup)
        return md

    async def cleanup(self) -> None:
        for closer in (self.conversation, self.workflow_runner, self.execution,
                       self.memory, self.bus):
            try:
                await closer.stop()
            except Exception:
                pass
        self.environment.cleanup()


class QualificationServiceFactory:
    """Builds a fresh isolated real service bundle per (task, action)."""

    def __init__(self, config: QualificationConfig, *, sandbox_root: str) -> None:
        self.config = config
        self.sandbox_root = Path(sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    async def build(self, task: dict[str, Any]) -> QualificationServices:
        self._counter += 1
        env_id = f"cf_{task['task_id']}_{self._counter}_{uuid.uuid4().hex[:10]}"
        root = self.sandbox_root / env_id
        env = IsolatedEnvironment(
            env_id=env_id,
            root=root,
            memory_db=root / "memory.sqlite",
            experience_db=root / "experience.sqlite",
            workflow_db=root / "workflow.sqlite",
            conversation_db=root / "conversation.sqlite",
            fs_root=root / "fs",
            exec_dir=root / "exec",
            seed=self._counter,
        )
        root.mkdir(parents=True, exist_ok=True)
        env.fs_root.mkdir(parents=True, exist_ok=True)
        env.exec_dir.mkdir(parents=True, exist_ok=True)

        bus = LocalEventBus()
        await bus.start()
        memory = MemoryService(bus, cfg={"db_path": str(env.memory_db)})
        await memory.start()

        provider, provider_name = self._make_provider()
        gateway = LLMGateway(
            cfg={
                "enable": True,
                "default_model": self.config.model_id,
                "models": {
                    self.config.model_id: {
                        "provider": provider_name,
                        "model_name": self.config.model_id,
                        "capabilities": ["text", "tools"],
                    }
                },
            },
            providers={provider_name: provider},
        )
        gateway.state = ServiceState.RUNNING

        tool_registry = ToolRegistry(cfg={"allow_side_effects": True})
        await tool_registry.start()
        execution = ExecutionService(
            bus,
            cfg={
                "allow": True,
                "unsafe_allow_host_execution": True,
                "timeout_s": 15.0,
                "max_output_chars": 20000,
                "allowed_commands": ["python"],
                "cwd_root": str(env.exec_dir),
            },
        )
        await execution.start()
        workflow_runner = WorkflowRunnerService(
            bus, cfg={"db_path": str(env.workflow_db), "auto_resume": False}
        )
        await workflow_runner.start()

        registry = ServiceRegistry()
        registry.register(bus)
        registry.register(memory)
        registry.register(gateway)
        registry.register(tool_registry)
        registry.register(execution)
        registry.register(workflow_runner)

        default_wf = build_plan_generate_test_reflect_workflow(
            services=registry,
            max_iterations=2,
            require_approval=False,
            verification_mode="acceptance",
            allow_smoke_success=False,
        )
        workflow_runner.register_workflow(default_wf)

        conversation = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gateway,
            cfg={
                "db_path": str(env.conversation_db),
                "use_tools": True,
                "max_tool_iterations": 3,
            },
            tool_registry=tool_registry,
            workflow_runner=workflow_runner,
            autolearn_enabled=False,
        )
        await conversation.start()

        controller = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=_ConversationActionDispatcher(conversation),
            experience_sink=MemoryExperienceSink(),
        )

        provisioned = await self._provision(task, memory, tool_registry, env)

        return QualificationServices(
            environment=env,
            bus=bus,
            memory=memory,
            tool_registry=tool_registry,
            workflow_runner=workflow_runner,
            execution=execution,
            conversation=conversation,
            controller=controller,
            registry=registry,
            provisioned=provisioned,
            provider=provider,
        )

    def _make_provider(self) -> tuple[Any, str]:
        if self.config.provider == PROVIDER_LOCAL_TRANSFORMERS:
            from .local_transformers_provider import LocalTransformersProvider
            p = LocalTransformersProvider(
                model_id=self.config.model,
                device=self.config.device,
                dtype=self.config.dtype,
            )
            return p, p.name
        p = GroundedQualificationProvider()
        return p, p.name

    async def _provision(
        self,
        task: dict[str, Any],
        memory: MemoryService,
        tool_registry: ToolRegistry,
        env: IsolatedEnvironment,
    ) -> ProvisionedTask:
        family = task["family"]
        setup = dict(task.get("setup_spec", {}) or {})

        if family == "memory_required":
            secret = setup.get("secret", f"MEM-{_secrets.token_hex(8)}")
            key = setup.get("key", "recall")
            await memory.write(
                f"{key}={secret}",
                type=MemoryType.OBSERVATION,
                source="provisioner",
                conversation_id=env.env_id,
                metadata={"key": key, "verified": True},
            )
            if setup.get("conflict_secret"):
                await memory.write(
                    f"{key}={setup['conflict_secret']}",
                    type=MemoryType.OBSERVATION,
                    source="provisioner_decoy",
                    conversation_id=env.env_id,
                    metadata={"key": key, "verified": False},
                )
            if setup.get("stale_secret"):
                await memory.write(
                    f"{key}={setup['stale_secret']}",
                    type=MemoryType.OBSERVATION,
                    source="provisioner_stale",
                    conversation_id=env.env_id,
                    metadata={"key": key, "verified": False, "stale": True},
                )
            return ProvisionedTask(
                task_id=task["task_id"],
                family=family,
                verifier_state={"expected_secret": secret},
                expected_output=secret,
            )

        if family == "tool_required":
            tool_name = setup.get("tool_name", "bench_tool")
            expected = setup.get("expected_tool_output", f"TOOL-{_secrets.token_hex(8)}")
            invocation_counter = {"count": 0}

            async def _handler(args: dict[str, Any]) -> str:
                invocation_counter["count"] += 1
                return expected

            spec = ToolSpec(
                name=tool_name,
                description="Benchmark tool returning the current value.",
                parameters={"type": "object", "properties": {}},
                side_effecting=False,
            )
            tool_registry.register(spec, _handler)
            return ProvisionedTask(
                task_id=task["task_id"],
                family=family,
                verifier_state={
                    "expected_tool_output": expected,
                    "tool_name": tool_name,
                    "expected_invocation_count": 1,
                    "invocation_counter": invocation_counter,
                },
                expected_output=expected,
            )

        if family == "workflow_required":
            return ProvisionedTask(
                task_id=task["task_id"],
                family=family,
                verifier_state={
                    "expected_marker": (task.get("verifier_spec", {}) or {}).get("expected_marker", "ACCEPTANCE_OK"),
                    "nonce": setup.get("nonce", ""),
                },
                expected_output="ACCEPTANCE_OK",
            )

        if family == "safety_adversarial":
            return ProvisionedTask(
                task_id=task["task_id"],
                family=family,
                verifier_state={"requires_safety": True},
                expected_output="blocked",
            )

        # direct_answer
        return ProvisionedTask(
            task_id=task["task_id"],
            family=family,
            verifier_state={"expected_token": setup.get("expected_token", "")},
            expected_output=setup.get("expected_token", ""),
        )
