from __future__ import annotations

import logging
from typing import Any

from capsule_brain.autolearn.controller import (
    ActionDispatcher,
    DispatcherResult,
    ExecutiveState,
)
from capsule_brain.autolearn.schema import Action
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .models import AssistantResponse, RequestContext, Turn, TurnRole
from .repository import ConversationRepository

log = logging.getLogger(__name__)


class ConversationService(CapsuleService):
    name = "conversation"

    def __init__(
        self,
        event_bus: LocalEventBus,
        memory: MemoryService,
        llm: LLMGateway,
        cfg: dict[str, Any] | None = None,
        repository: ConversationRepository | None = None,
        experience_store=None,
        tool_registry=None,
        autolearn_service=None,
        workflow_runner=None,
        autolearn_enabled: bool | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.memory = memory
        self.llm = llm
        self.experience_store = experience_store
        self.tool_registry = tool_registry
        self.autolearn_service = autolearn_service
        self.workflow_runner = workflow_runner
        # Native tool use is opt-in because not every configured model route
        # advertises the tools capability. When enabled, an empty registry
        # falls back to plain generation; tools registered later become
        # available without rewiring the conversation service.
        self.use_tools = bool(self.cfg.get("use_tools", False))
        self.max_tool_iterations = max(1, int(self.cfg.get("max_tool_iterations", 4)))
        self.repository = repository or ConversationRepository(
            self.cfg.get("db_path", "data/conversations_v2.sqlite")
        )
        self.route = self.cfg.get("route", "chat")
        self.model = self.cfg.get("model")
        self.history_limit = int(self.cfg.get("history_limit", 12))
        self.memory_limit = int(self.cfg.get("memory_limit", 8))
        # Semantic memory retrieval. When enabled, the conversation service
        # embeds the user's message and retrieves the top-K semantically
        # relevant memories instead of the most recent K. Falls back to
        # chronological retrieval if semantic search is unavailable.
        self.semantic_memory = bool(self.cfg.get("semantic_memory", False))
        self.semantic_memory_limit = int(
            self.cfg.get("semantic_memory_limit", self.memory_limit)
        )
        self.system_prompt = self.cfg.get(
            "system_prompt",
            "You are Capsule Brain, a precise autonomous AI assistant.",
        )
        # AutoLearn v0.3: the enable flag is passed explicitly from
        # build_application() (derived from autolearn.enable), NOT parsed
        # from a second conversation.autolearn_enable flag. This eliminates
        # the silent bypass bug where autolearn.enable=true but
        # conversation.autolearn_enable was unset.
        # autolearn_enabled defaults to False when autolearn_service is None.
        if autolearn_enabled is not None:
            self.autolearn_enabled = bool(autolearn_enabled)
        else:
            self.autolearn_enabled = bool(autolearn_service is not None)
        self._unsubscribers: list = []
        self._requests = 0
        self._failures = 0
        # Build the action dispatcher that delegates to our internal
        # _respond_* methods. Attached to the ExecutiveController when
        # AutoLearn is enabled.
        self._dispatcher = _ConversationActionDispatcher(self)
        # v0.3.2: No mutable per-request field. RequestContext is built
        # per call and threaded explicitly through execute_executive_action
        # / _respond_via_autolearn into the _respond_* methods. This makes
        # concurrent request handling safe — no shared mutable state.

    # ------------------------------------------------------------------
    # Public executive-action API (v0.3)
    # ------------------------------------------------------------------
    #
    # This is the single public entry point for dispatching executive
    # actions through real Capsule Brain services. Both the
    # ExecutiveController (production path) and RealCounterfactualRuntime
    # (qualification path) use this same method. There is only one
    # implementation of runtime action semantics.
    #
    # Supported actions: ANSWER_DIRECT, RETRIEVE_MEMORY, CALL_TOOL,
    # START_WORKFLOW. REFLECT and ASK_OPERATOR are handled by the baseline
    # safety path and are not dispatched through this API.

    async def execute_executive_action(
        self,
        *,
        action: "Action",
        state: "ExecutiveState",
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
        request_context: RequestContext | None = None,
    ) -> "DispatcherResult":
        """Execute one executive action through real Capsule Brain services.

        This is the canonical action dispatcher. It builds an immutable
        ``RequestContext`` from ``state`` and ``metadata`` (or reuses a
        caller-supplied one), delegates to the corresponding ``_respond_*``
        method, and returns a ``DispatcherResult``. Both production routing
        (via ExecutiveController) and counterfactual qualification (via
        RealCounterfactualRuntime) call this same method.

        v0.3.2: The request context is passed explicitly to the
        ``_respond_*`` methods. No mutable service-level request field is
        used, so concurrent requests cannot observe each other's state.

        Args:
            action: One of ANSWER_DIRECT, RETRIEVE_MEMORY, CALL_TOOL,
                START_WORKFLOW.
            state: The executive state (prompt, memory, tools, etc.).
            conversation_id: Unique conversation ID for this execution.
            metadata: Optional metadata (e.g. task_id, counterfactual
                context). When ``metadata`` contains ``text``, ``history``,
                ``memory_records``, they are used as the request context;
                otherwise the context is built from ``state``.
            request_context: Optional pre-built immutable context. When
                provided, it is used directly and ``metadata`` is ignored
                for context construction.

        Returns:
            DispatcherResult with the real outcome of the action.
        """
        from capsule_brain.autolearn.schema import Action as _Action

        # Build per-request context from state + metadata, or reuse a
        # caller-supplied immutable context.
        if request_context is None:
            md = metadata or {}
            text = md.get("text") or str(state.prompt_features.get("text", ""))
            history = md.get("history", [])
            memory_records = md.get("memory_records", [])

            # If no history is provided, include the user's text as the first
            # turn so the LLM provider can see the prompt. Without this, the
            # _build_context() method produces an empty context and the provider
            # has no idea what the user asked.
            if not history and text:
                from .models import Turn, TurnRole
                history = [Turn(role=TurnRole.USER, text=text)]

            request_context = RequestContext(
                request_id=md.get("request_id", f"cf_{conversation_id}"),
                conversation_id=conversation_id,
                prompt=text,
                history=tuple(history),
                memory_records=tuple(memory_records),
                available_tools=tuple(state.available_tools),
                correlation_id=md.get("correlation_id"),
                user_turn_id=md.get("user_turn_id", "cf_turn"),
                metadata=dict(md),
            )

        if action == _Action.ANSWER_DIRECT:
            return await self._respond_direct(
                state, conversation_id=conversation_id, request_context=request_context
            )
        elif action == _Action.RETRIEVE_MEMORY:
            return await self._respond_with_memory(
                state, conversation_id=conversation_id, request_context=request_context
            )
        elif action == _Action.CALL_TOOL:
            return await self._respond_with_tools(
                state, conversation_id=conversation_id, request_context=request_context
            )
        elif action == _Action.START_WORKFLOW:
            return await self._respond_with_workflow(
                state, conversation_id=conversation_id, request_context=request_context
            )
        else:
            return DispatcherResult(
                runtime_error=True,
                action_completed=False,
                metadata={"reason": f"unsupported action for execute_executive_action: {action}"},
            )

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        if self.experience_store is not None:
            await self.experience_store.start()
        # Wire the ExecutiveController's dispatcher to this conversation
        # service so the controller can dispatch actions through our real
        # LLMGateway / MemoryService / ToolRegistry / WorkflowRunner paths.
        if self.autolearn_enabled and self.autolearn_service is not None:
            self.autolearn_service.attach_dispatcher(self._dispatcher)
        self._unsubscribers.append(
            self.event_bus.subscribe(
                "conversation.message.create",
                self._on_message_create,
            )
        )
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def _on_message_create(self, event: EventEnvelope) -> None:
        payload = event.payload
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        conversation = await self.repository.ensure_conversation(
            payload.get("conversation_id")
        )
        try:
            await self.respond(
                conversation_id=conversation.id,
                text=text,
                correlation_id=str(event.correlation_id)
                if event.correlation_id else None,
            )
        except Exception as exc:
            self._failures += 1
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="conversation.response.failed",
                    source=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "conversation_id": conversation.id,
                        "error": str(exc),
                    },
                )
            )

    async def respond(
        self,
        *,
        conversation_id: str,
        text: str,
        correlation_id: str | None = None,
    ) -> AssistantResponse:
        self._requests += 1
        conversation = await self.repository.ensure_conversation(conversation_id)

        user_turn = Turn(
            conversation_id=conversation.id,
            role=TurnRole.USER,
            text=text,
        )
        await self.repository.create_turn(user_turn)

        # Retrieve long-term context before indexing the current turn so the
        # query cannot retrieve itself and waste a semantic top-K slot. The
        # current user message is already present in conversation history.
        memory_records = await self._retrieve_memories(text)

        await self.memory.write(
            text,
            type=MemoryType.OPERATOR,
            source="operator",
            conversation_id=conversation.id,
            turn_id=user_turn.id,
            protected=True,
            metadata={"conversation_role": "user"},
        )

        history = await self.repository.recent_turns(
            conversation.id,
            limit=self.history_limit,
        )

        # ------------------------------------------------------------------
        # AutoLearn v0.2 path: route through the ExecutiveController, which
        # selects an action and dispatches it through real services via the
        # _ConversationActionDispatcher. The dispatcher calls one of the
        # _respond_* methods below. When AutoLearn is disabled, the original
        # direct-generation path runs unchanged (byte-for-byte compatible).
        # ------------------------------------------------------------------
        if self.autolearn_enabled and self.autolearn_service is not None:
            return await self._respond_via_autolearn(
                conversation=conversation,
                user_turn=user_turn,
                text=text,
                history=history,
                memory_records=memory_records,
                correlation_id=correlation_id,
            )

        # Legacy direct-generation path (preserved exactly).
        context = self._build_context(history, memory_records)

        llm_request = LLMRequest(
            prompt=context,
            system=self.system_prompt,
            model=self.model,
            metadata={
                "conversation_id": conversation.id,
                "parent_turn_id": user_turn.id,
            },
        )
        tool_results = []
        if (
            self.use_tools
            and self.tool_registry is not None
            and self.tool_registry.specs()
        ):
            result, tool_results = await self.llm.generate_with_tools(
                llm_request,
                self.tool_registry,
                route=self.route,
                max_iterations=self.max_tool_iterations,
            )
        else:
            result = await self.llm.generate(
                llm_request,
                route=self.route,
            )

        return await self._finalize_response(
            conversation=conversation,
            user_turn=user_turn,
            text=text,
            result_text=result.text,
            result=result,
            tool_results=tool_results,
        )

    async def _respond_via_autolearn(
        self,
        *,
        conversation,
        user_turn,
        text: str,
        history,
        memory_records,
        correlation_id: str | None,
    ) -> AssistantResponse:
        """Route the request through the ExecutiveController.

        Builds an ExecutiveState from the current runtime context, calls
        ``autolearn_service.execute()``, which selects an action and
        dispatches it through the ``_ConversationActionDispatcher`` to one of
        the real ``_respond_*`` methods. The dispatcher result is then
        finalized into an AssistantResponse.
        """
        # Build ExecutiveState from real runtime context. Features describe
        # state only — never the answer.
        memory_hit_count = len(memory_records) if memory_records else 0
        top_similarity = 0.0
        if memory_records:
            for rec in memory_records:
                sim = getattr(rec, "similarity", None)
                if sim is not None and sim > top_similarity:
                    top_similarity = float(sim)
        available_tools = (
            [spec.name for spec in self.tool_registry.specs()]
            if self.tool_registry is not None
            else []
        )
        workflow_available = self.workflow_runner is not None
        state = ExecutiveState(
            prompt_features={
                "text": text,
                "previous_attempt_failed": False,
                "verification_failure_type": "none",
                "estimated_difficulty": 0.5,
                "workflow_capability_match": workflow_available,
            },
            conversation_features={"depth": len(history)},
            memory_features={
                "hit_count": memory_hit_count,
                "top_similarity": top_similarity,
            },
            available_tools=available_tools,
            workflow_available=workflow_available,
            model_id=self.model,
            context_length=len(text),
        )

        # v0.3.2: Build an immutable RequestContext and thread it explicitly
        # through autolearn_service.execute -> controller -> dispatcher ->
        # _respond_*. No mutable service-level field is used.
        request_context = RequestContext(
            request_id=f"prod_{conversation.id}_{user_turn.id}",
            conversation_id=conversation.id,
            prompt=text,
            history=tuple(history),
            memory_records=tuple(memory_records),
            available_tools=tuple(available_tools),
            correlation_id=correlation_id,
            user_turn_id=user_turn.id,
        )
        action_result = await self.autolearn_service.execute(
            state,
            conversation_id=conversation.id,
            task_family="production",
            request_context=request_context,
        )

        # Finalize the dispatched result into an AssistantResponse.
        # Reconstruct a minimal LLMResult-like object for finalize.
        result_text = action_result.text
        return await self._finalize_response(
            conversation=conversation,
            user_turn=user_turn,
            text=text,
            result_text=result_text,
            result=None,
            tool_results=[],
            autolearn_action=action_result.action.value,
            autolearn_outcome=action_result.outcome,
        )

    # ------------------------------------------------------------------
    # Internal _respond_* methods — each executes one action through real
    # services and returns a DispatcherResult. Called by the
    # _ConversationActionDispatcher (which the ExecutiveController uses).
    # ------------------------------------------------------------------

    def _current_llm_request(
        self,
        context_override: str | None = None,
        *,
        request_context: RequestContext | None = None,
    ) -> LLMRequest:
        """Build an LLMRequest from an explicit request context.

        v0.3.2: Reads from the passed-in immutable ``request_context`` rather
        than a mutable service-level field.
        """
        ctx = request_context
        history = list(ctx.history) if ctx else []
        memory_records = list(ctx.memory_records) if ctx else []
        conversation_id = ctx.conversation_id if ctx else ""
        user_turn_id = ctx.user_turn_id if ctx else ""
        context = context_override or self._build_context(history, memory_records)
        return LLMRequest(
            prompt=context,
            system=self.system_prompt,
            model=self.model,
            metadata={
                "conversation_id": conversation_id,
                "parent_turn_id": user_turn_id,
            },
        )

    async def _respond_direct(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: RequestContext | None = None,
    ) -> DispatcherResult:
        """ANSWER_DIRECT: LLM generation with no tools and no memory context."""
        import time as _time
        started = _time.perf_counter()
        # Direct answer uses only conversation history, no memory context.
        ctx = request_context
        history = list(ctx.history) if ctx else []
        context = self._build_context(history, [])
        llm_request = self._current_llm_request(
            context_override=context, request_context=request_context
        )
        result = await self.llm.generate(llm_request, route=self.route)
        latency = (_time.perf_counter() - started) * 1000.0
        tokens = sum(result.usage.values()) if result.usage else max(20, len(result.text) // 4)
        return DispatcherResult(
            text=result.text,
            latency_ms=latency,
            token_count=int(tokens),
            action_completed=True,
            metadata={"model": result.model, "provider": result.provider},
        )

    async def _respond_with_memory(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: RequestContext | None = None,
    ) -> DispatcherResult:
        """RETRIEVE_MEMORY: real MemoryService retrieval + LLM generation with evidence."""
        import time as _time
        started = _time.perf_counter()
        ctx = request_context
        text = ctx.prompt if ctx else ""
        # Real memory retrieval through MemoryService.
        memory_records = await self._retrieve_memories(text)
        history = list(ctx.history) if ctx else []
        context = self._build_context(history, memory_records)
        llm_request = self._current_llm_request(
            context_override=context, request_context=request_context
        )
        result = await self.llm.generate(llm_request, route=self.route)
        latency = (_time.perf_counter() - started) * 1000.0
        tokens = sum(result.usage.values()) if result.usage else max(30, len(result.text) // 4)
        return DispatcherResult(
            text=result.text,
            latency_ms=latency,
            token_count=int(tokens),
            action_completed=True,
            metadata={
                "model": result.model,
                "provider": result.provider,
                "memory_records_retrieved": len(memory_records) if memory_records else 0,
            },
        )

    async def _respond_with_tools(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: RequestContext | None = None,
    ) -> DispatcherResult:
        """CALL_TOOL: real LLMGateway.generate_with_tools() through ToolRegistry."""
        import time as _time
        started = _time.perf_counter()
        if self.tool_registry is None or not self.tool_registry.specs():
            return DispatcherResult(
                runtime_error=True,
                action_completed=False,
                metadata={"reason": "no tools registered"},
            )
        ctx = request_context
        history = list(ctx.history) if ctx else []
        memory_records = list(ctx.memory_records) if ctx else []
        context = self._build_context(history, memory_records)
        llm_request = self._current_llm_request(
            context_override=context, request_context=request_context
        )
        result, tool_results = await self.llm.generate_with_tools(
            llm_request,
            self.tool_registry,
            route=self.route,
            max_iterations=self.max_tool_iterations,
        )
        latency = (_time.perf_counter() - started) * 1000.0
        tokens = sum(result.usage.values()) if result.usage else max(30, len(result.text) // 4)
        tool_failures = sum(1 for item in tool_results if item.is_error)
        tool_names = [item.name for item in tool_results]
        # Determine if any tool result is valid (non-error).
        tool_result_valid = any(not item.is_error for item in tool_results) if tool_results else False
        return DispatcherResult(
            text=result.text,
            latency_ms=latency,
            token_count=int(tokens),
            tool_failures=tool_failures,
            action_completed=True,
            tool_name=tool_names[0] if tool_names else None,
            tool_calls_executed=len(tool_results),
            tool_result_valid=tool_result_valid,
            metadata={
                "model": result.model,
                "provider": result.provider,
                "tool_names": tool_names,
            },
        )

    async def _respond_with_workflow(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: RequestContext | None = None,
    ) -> DispatcherResult:
        """START_WORKFLOW: real WorkflowRunnerService.start_run()."""
        import time as _time
        started = _time.perf_counter()
        if self.workflow_runner is None:
            return DispatcherResult(
                runtime_error=True,
                action_completed=False,
                metadata={"reason": "no workflow runner available"},
            )
        ctx = request_context
        text = ctx.prompt if ctx else ""
        # Determine the workflow name. v0.2 uses the default
        # plan_generate_test_reflect workflow.
        workflow_name = "plan_generate_test_reflect"
        # v0.3.2: Pass an acceptance verification contract through to the
        # workflow when the request context carries one (qualification tasks).
        # The contract supplies the acceptance test file + command so the
        # workflow's acceptance verification is real, not inferred from
        # run.status == COMPLETED alone.
        run_metadata: dict[str, Any] = {"conversation_id": conversation_id or ""}
        if ctx is not None:
            md = ctx.metadata or {}
            acceptance_code = md.get("acceptance_code")
            if acceptance_code:
                run_metadata["verification"] = {
                    "files": {"acceptance_test.py": acceptance_code},
                    "command": ["python", "acceptance_test.py"],
                }
        try:
            run = await self.workflow_runner.start_run(
                workflow_name=workflow_name,
                goal=text,
                metadata=run_metadata,
            )
        except Exception as exc:
            return DispatcherResult(
                runtime_error=True,
                action_completed=False,
                metadata={"reason": f"workflow start failed: {exc}"},
            )
        latency = (_time.perf_counter() - started) * 1000.0
        # Extract the workflow outcome from the run. v0.3.2: derive
        # acceptance_passed from the actual acceptance test result stored on
        # the run state, not from run.status alone.
        from capsule_brain.workflow.models import WorkflowStatus
        state_extra = run.state.extra if run.state else {}
        acceptance_present = bool(state_extra.get("verification_contract_present", False)) or (
            run.status == WorkflowStatus.COMPLETED
        )
        acceptance_passed = bool(state_extra.get("test_passed", False))
        exit_code = state_extra.get("exit_code")
        if exit_code is None:
            exit_code = 0 if acceptance_passed else 1
        workflow_iterations = run.state.iterations if run.state else 0
        code = run.state.code if run.state else ""
        return DispatcherResult(
            text=code or run.stop_reason or "",
            latency_ms=latency,
            token_count=max(120, len(code) // 3) if code else 200,
            workflow_iterations=workflow_iterations,
            action_completed=True,
            workflow_name=workflow_name,
            workflow_run_id=run.id,
            acceptance_present=acceptance_present,
            acceptance_passed=acceptance_passed,
            exit_code=exit_code,
            metadata={
                "run_status": run.status.value if run.status else "unknown",
                "stop_reason": run.stop_reason,
                "verification_kind": state_extra.get("verification_kind", ""),
                "acceptance_contract_present": acceptance_present,
                "test_passed": acceptance_passed,
                "exit_code": exit_code,
                "stdout_digest": "",
                "stderr_digest": "",
                "artifact_paths": [],
                "required_artifact_exists": acceptance_passed,
                "reflection_count": workflow_iterations,
                "execution_attempts": workflow_iterations,
                "workflow_evidence": {
                    "workflow_run_id": run.id,
                    "workflow_status": run.status.value if run.status else "unknown",
                    "verification_kind": state_extra.get("verification_kind", ""),
                    "acceptance_contract_present": acceptance_present,
                    "test_passed": acceptance_passed,
                    "exit_code": exit_code,
                    "required_artifact_exists": acceptance_passed,
                    "reflection_count": workflow_iterations,
                    "execution_attempts": workflow_iterations,
                },
            },
        )

    async def _finalize_response(
        self,
        *,
        conversation,
        user_turn,
        text: str,
        result_text: str,
        result,
        tool_results: list,
        autolearn_action: str | None = None,
        autolearn_outcome=None,
    ) -> AssistantResponse:
        """Common postamble: create assistant turn, response, experience, events."""
        assistant_turn = Turn(
            conversation_id=conversation.id,
            role=TurnRole.ASSISTANT,
            text=result_text,
            parent_turn_id=user_turn.id,
        )
        await self.repository.create_turn(assistant_turn)

        if result is not None:
            trace = result.trace
            model_alias = trace.model_alias if trace else self.model
            route_name = trace.route if trace else self.route
            provider = result.provider
            model_name = result.model
            latency_ms = result.latency_ms
            attempts = result.attempts
            usage = dict(result.usage)
        else:
            # Autolearn path: synthesize from the outcome.
            model_alias = self.model
            route_name = self.route
            provider = "autolearn"
            model_name = self.model or "unknown"
            latency_ms = autolearn_outcome.latency_ms if autolearn_outcome else 0.0
            attempts = 1
            usage = {}

        response = AssistantResponse(
            conversation_id=conversation.id,
            parent_turn_id=user_turn.id,
            turn_id=assistant_turn.id,
            text=result_text,
            model_alias=model_alias,
            provider=provider,
            model_name=model_name,
            latency_ms=latency_ms,
            attempts=attempts,
            usage=usage,
            route=route_name,
        )
        await self.repository.create_response(response)

        if self.experience_store is not None:
            from capsule_brain.learning.models import ExperienceRecord
            await self.experience_store.create_experience(
                ExperienceRecord(
                    response_id=response.id,
                    conversation_id=response.conversation_id,
                    user_turn_id=response.parent_turn_id,
                    assistant_turn_id=response.turn_id,
                    prompt_text=text,
                    response_text=response.text,
                    model_alias=response.model_alias,
                    provider=response.provider,
                    model_name=response.model_name,
                    route=response.route,
                    latency_ms=response.latency_ms,
                    attempts=response.attempts,
                    usage=dict(response.usage),
                    metadata={
                        "tool_calls_executed": len(tool_results),
                        "tools_used": [item.name for item in tool_results],
                        "tool_failures": sum(
                            1 for item in tool_results if item.is_error
                        ),
                        "autolearn_action": autolearn_action,
                    },
                )
            )

        await self.memory.write(
            result_text,
            type=MemoryType.ASSISTANT,
            source="assistant",
            conversation_id=conversation.id,
            turn_id=assistant_turn.id,
            metadata={
                "response_id": response.id,
                "parent_turn_id": user_turn.id,
                "model_alias": response.model_alias,
                "provider": response.provider,
                "model_name": response.model_name,
                "route": response.route,
            },
        )

        await self.event_bus.publish(
            EventEnvelope(
                event_type="conversation.response.created",
                source=self.name,
                payload={
                    "response_id": response.id,
                    "conversation_id": response.conversation_id,
                    "parent_turn_id": response.parent_turn_id,
                    "turn_id": response.turn_id,
                    "text": response.text,
                    "model_alias": response.model_alias,
                    "provider": response.provider,
                    "model_name": response.model_name,
                    "latency_ms": response.latency_ms,
                    "attempts": response.attempts,
                    "usage": response.usage,
                    "route": response.route,
                    "tool_calls_executed": len(tool_results),
                    "tools_used": [item.name for item in tool_results],
                    "autolearn_action": autolearn_action,
                },
            )
        )

        # Compatibility event for GUI layers that only need text.
        await self.event_bus.publish(
            EventEnvelope(
                event_type="gui.response",
                source=self.name,
                payload={
                    "conversation_id": response.conversation_id,
                    "response_id": response.id,
                    "text": response.text,
                },
            )
        )

        return response

    async def _retrieve_memories(self, user_text: str):
        """Retrieve memories relevant to ``user_text``.

        When semantic_memory is enabled and the memory service has a working
        embedding provider, embeds the user message and retrieves the top-K
        semantically similar memories. Falls back to chronological recent()
        retrieval on any error so a transient embedding failure never breaks
        the conversation path.
        """
        if self.semantic_memory:
            try:
                return await self.memory.search_semantic(
                    user_text,
                    limit=self.semantic_memory_limit,
                    include_archived=False,
                )
            except Exception:
                pass
        return await self.memory.recent(
            limit=self.memory_limit,
            include_archived=False,
        )

    def _build_context(self, history, memories) -> str:
        parts = ["Conversation history:"]
        for turn in history:
            parts.append(f"{turn.role.value.upper()}: {turn.text}")

        if memories:
            parts.append("\nRelevant recent memory:")
            seen: set[str] = set()
            for memory in memories:
                # Skip OPERATOR and ASSISTANT turn memories — their text
                # already appears in the Conversation history section above.
                # Including them here would duplicate context and waste tokens.
                if memory.type in {MemoryType.OPERATOR, MemoryType.ASSISTANT}:
                    continue
                key = memory.text.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                parts.append(f"- [{memory.type.value}] {memory.text}")

        parts.append("\nRespond to the latest USER message.")
        return "\n".join(parts)

    async def health(self) -> HealthStatus:
        state = (
            ServiceState.DEGRADED
            if self.state == ServiceState.RUNNING and self._failures
            else self.state
        )
        return HealthStatus(
            state=state,
            details={
                "requests": self._requests,
                "failures": self._failures,
                "db_path": str(self.repository.db_path),
                "route": self.route,
                "model": self.model,
                "use_tools": self.use_tools,
                "registered_tools": (
                    [spec.name for spec in self.tool_registry.specs()]
                    if self.tool_registry is not None
                    else []
                ),
                "max_tool_iterations": self.max_tool_iterations,
                "autolearn_enabled": self.autolearn_enabled,
            },
        )


class _ConversationActionDispatcher(ActionDispatcher):
    """Dispatches executive actions through the real ConversationService paths.

    Each method delegates to the corresponding ``_respond_*`` internal method
    on the owning ConversationService. This is the concrete bridge between
    the ExecutiveController and the real Capsule Brain service stack.
    """

    def __init__(self, service: ConversationService) -> None:
        self._service = service

    async def answer_direct(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        return await self._service._respond_direct(
            state, conversation_id=conversation_id, request_context=request_context
        )

    async def retrieve_memory(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        return await self._service._respond_with_memory(
            state, conversation_id=conversation_id, request_context=request_context
        )

    async def call_tool(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        return await self._service._respond_with_tools(
            state, conversation_id=conversation_id, request_context=request_context
        )

    async def start_workflow(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        return await self._service._respond_with_workflow(
            state, conversation_id=conversation_id, request_context=request_context
        )
