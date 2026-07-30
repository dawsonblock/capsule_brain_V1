"""Tests for immutable RequestContext isolation (v0.3.2 / 2.15.3 Section 1).

Verifies that:
- RequestContext is immutable (frozen dataclass).
- Concurrent requests do not observe each other's state.
- No mutable service-level request context remains in production code.
- The real ConversationService threads request context explicitly and
  concurrent requests with different prompts cannot cross-contaminate.
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from capsule_brain.conversation.models import RequestContext, Turn, TurnRole


class TestRequestContextImmutability:
    """Test that RequestContext is immutable."""

    def test_request_context_is_frozen(self):
        ctx = RequestContext(
            request_id="r1",
            conversation_id="c1",
            prompt="hello",
        )
        with pytest.raises(Exception):
            ctx.prompt = "hacked"

    def test_request_context_history_is_tuple(self):
        turn = Turn(role=TurnRole.USER, text="hi")
        ctx = RequestContext(
            request_id="r1",
            conversation_id="c1",
            prompt="hello",
            history=(turn,),
        )
        assert isinstance(ctx.history, tuple)
        assert len(ctx.history) == 1

    def test_request_context_memory_records_is_tuple(self):
        ctx = RequestContext(
            request_id="r1",
            conversation_id="c1",
            prompt="hello",
            memory_records=("mem1",),
        )
        assert isinstance(ctx.memory_records, tuple)

    def test_request_context_available_tools_is_tuple(self):
        ctx = RequestContext(
            request_id="r1",
            conversation_id="c1",
            prompt="hello",
            available_tools=("tool1",),
        )
        assert isinstance(ctx.available_tools, tuple)

    def test_two_contexts_are_independent(self):
        ctx_a = RequestContext(
            request_id="a",
            conversation_id="c1",
            prompt="prompt A",
        )
        ctx_b = RequestContext(
            request_id="b",
            conversation_id="c2",
            prompt="prompt B",
        )
        assert ctx_a.prompt != ctx_b.prompt
        assert ctx_a.request_id != ctx_b.request_id


class TestConcurrentRequestIsolation:
    """Test that concurrent requests remain isolated."""

    def test_concurrent_contexts_do_not_cross(self):
        """Simulate two concurrent requests building their own contexts."""
        async def _run():
            results = {}

            async def _request(label: str, prompt: str):
                ctx = RequestContext(
                    request_id=label,
                    conversation_id=f"conv_{label}",
                    prompt=prompt,
                )
                # Simulate some async work.
                await asyncio.sleep(0.001)
                # Verify the context was not overwritten.
                results[label] = ctx.prompt

            await asyncio.gather(
                _request("A", "prompt A"),
                _request("B", "prompt B"),
            )
            assert results["A"] == "prompt A"
            assert results["B"] == "prompt B"

        asyncio.run(_run())

    def test_concurrent_contexts_have_different_conversation_ids(self):
        async def _run():
            contexts = []

            async def _make(cid: str):
                ctx = RequestContext(
                    request_id=f"r_{cid}",
                    conversation_id=cid,
                    prompt=f"prompt_{cid}",
                )
                await asyncio.sleep(0.001)
                contexts.append(ctx)

            await asyncio.gather(
                _make("conv1"),
                _make("conv2"),
                _make("conv3"),
            )
            cids = {c.conversation_id for c in contexts}
            assert cids == {"conv1", "conv2", "conv3"}

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# v0.3.2: production-code hygiene + real ConversationService isolation
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class TestNoMutableRequestContextInProduction:
    """Section 1: no self._current_request may remain in production code."""

    def test_no_current_request_field_in_conversation_service(self):
        from capsule_brain.conversation import service as svc_mod

        src = inspect.getsource(svc_mod)
        assert "_current_request" not in src, (
            "ConversationService must not reference _current_request (mutable "
            "shared request context). Use explicit RequestContext threading."
        )

    def test_no_current_request_in_any_production_module(self):
        import capsule_brain

        pkg_dir = Path(list(capsule_brain.__path__)[0]).resolve()
        offenders: list[str] = []
        for py in pkg_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            # Allow the models.py docstring to mention the historical name.
            if py.name == "models.py":
                continue
            if "_current_request" in text:
                offenders.append(str(py.relative_to(pkg_dir)))
        assert not offenders, (
            "Mutable _current_request found in production modules: "
            f"{offenders}"
        )

    def test_respond_methods_accept_request_context(self):
        from capsule_brain.conversation.service import ConversationService

        for name in (
            "_respond_direct",
            "_respond_with_memory",
            "_respond_with_tools",
            "_respond_with_workflow",
            "execute_executive_action",
        ):
            sig = inspect.signature(getattr(ConversationService, name))
            assert "request_context" in sig.parameters, (
                f"{name} must accept request_context explicitly"
            )

    def test_dispatcher_methods_accept_request_context(self):
        from capsule_brain.conversation.service import _ConversationActionDispatcher

        for name in ("answer_direct", "retrieve_memory", "call_tool", "start_workflow"):
            sig = inspect.signature(getattr(_ConversationActionDispatcher, name))
            assert "request_context" in sig.parameters, (
                f"{name} must accept request_context explicitly"
            )

    def test_controller_execute_accepts_request_context(self):
        from capsule_brain.autolearn.controller import ExecutiveController

        sig = inspect.signature(ExecutiveController.execute)
        assert "request_context" in sig.parameters


class TestRealConversationServiceConcurrentIsolation:
    """Section 1: concurrent requests through the real ConversationService
    cannot observe each other's prompt/memory/tool/workflow context."""

    @pytest.mark.asyncio
    async def test_concurrent_prompts_do_not_cross(self, tmp_path):
        from capsule_brain.autolearn.baseline import BaselinePolicyV2
        from capsule_brain.autolearn.controller import (
            ActionDispatcher,
            DispatcherResult,
            ExecutiveController,
            MemoryExperienceSink,
        )
        from capsule_brain.autolearn.schema import Action, ExecutiveState
        from capsule_brain.conversation.service import ConversationService
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.models import LLMResult
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.runtime.service import ServiceState

        seen_prompts: list[str] = []

        class _RecordingProvider(LLMProvider):
            name = "fake"

            async def generate(self, request, model_cfg):
                seen_prompts.append(request.prompt)
                return LLMResult(
                    text="ok",
                    model="fake",
                    provider="fake",
                    latency_ms=1.0,
                    usage={"total_tokens": 5},
                )

        class _RecordingDispatcher(ActionDispatcher):
            async def answer_direct(self, state, *, conversation_id=None, request_context=None):
                seen_prompts.append(request_context.prompt)
                return DispatcherResult(text="ok", action_completed=True)

            async def retrieve_memory(self, state, *, conversation_id=None, request_context=None):
                seen_prompts.append(request_context.prompt)
                return DispatcherResult(text="ok", action_completed=True)

            async def call_tool(self, state, *, conversation_id=None, request_context=None):
                return DispatcherResult(text="ok", action_completed=True)

            async def start_workflow(self, state, *, conversation_id=None, request_context=None):
                return DispatcherResult(text="ok", action_completed=True)

        bus = LocalEventBus()
        await bus.start()
        memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
        await memory.start()
        gw = LLMGateway(
            cfg={
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}
                },
            },
            providers={"fake": _RecordingProvider()},
        )
        gw.state = ServiceState.RUNNING

        svc = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gw,
            cfg={"db_path": str(tmp_path / "conv.sqlite")},
            autolearn_enabled=True,
        )
        # Attach a controller whose dispatcher records the prompt seen per
        # request context.
        from capsule_brain.autolearn.service import AutoLearnService

        autolearn = AutoLearnService(cfg={})
        autolearn.attach_dispatcher(_RecordingDispatcher())
        autolearn.attach_experience_sink(MemoryExperienceSink())
        svc.autolearn_service = autolearn
        await svc.start()

        # Two concurrent requests with distinct prompts.
        await asyncio.gather(
            svc.respond(conversation_id="cA", text="PROMPT_AAA"),
            svc.respond(conversation_id="cB", text="PROMPT_BBB"),
        )

        # Each prompt must appear; no request should have seen the other's
        # prompt as its own context.
        joined = "\n".join(seen_prompts)
        assert "PROMPT_AAA" in joined
        assert "PROMPT_BBB" in joined
        # No single context string should contain BOTH prompts (which would
        # indicate cross-contamination of the per-request context).
        for entry in seen_prompts:
            assert not ("PROMPT_AAA" in entry and "PROMPT_BBB" in entry), (
                f"cross-contaminated context: {entry!r}"
            )

        await svc.stop()
        await memory.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_memory_context_does_not_cross_requests(self, tmp_path):
        """Two concurrent execute_executive_action calls with different
        memory_records must each receive their own request_context."""
        from capsule_brain.autolearn.controller import DispatcherResult
        from capsule_brain.autolearn.schema import Action, ExecutiveState
        from capsule_brain.conversation.models import RequestContext
        from capsule_brain.conversation.service import ConversationService
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.runtime.service import ServiceState

        class _StubProvider(LLMProvider):
            name = "fake"

            async def generate(self, request, model_cfg):
                from capsule_brain.llm.models import LLMResult
                return LLMResult(text="ok", model="fake", provider="fake", latency_ms=1.0, usage={})

        bus = LocalEventBus()
        await bus.start()
        memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
        await memory.start()
        gw = LLMGateway(
            cfg={
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}
                },
            },
            providers={"fake": _StubProvider()},
        )
        gw.state = ServiceState.RUNNING
        svc = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gw,
            cfg={"db_path": str(tmp_path / "conv.sqlite")},
        )
        await svc.start()

        # Record the request_context each _respond_with_memory call receives.
        received: list[RequestContext] = []
        original = svc._respond_with_memory

        async def _recording(state, *, conversation_id=None, request_context=None):
            received.append(request_context)
            return DispatcherResult(text="ok", action_completed=True)

        svc._respond_with_memory = _recording  # type: ignore[assignment]

        state = ExecutiveState(prompt_features={"text": "q"}, available_tools=[])
        ctx_a = RequestContext(
            request_id="ra", conversation_id="cA", prompt="qA", memory_records=("MEM_A",)
        )
        ctx_b = RequestContext(
            request_id="rb", conversation_id="cB", prompt="qB", memory_records=("MEM_B",)
        )

        await asyncio.gather(
            svc.execute_executive_action(
                action=Action.RETRIEVE_MEMORY,
                state=state,
                conversation_id="cA",
                request_context=ctx_a,
            ),
            svc.execute_executive_action(
                action=Action.RETRIEVE_MEMORY,
                state=state,
                conversation_id="cB",
                request_context=ctx_b,
            ),
        )
        svc._respond_with_memory = original  # type: ignore[assignment]

        assert len(received) == 2
        prompts = {ctx.prompt for ctx in received}
        mems = {ctx.memory_records for ctx in received}
        assert prompts == {"qA", "qB"}
        assert ("MEM_A",) in mems
        assert ("MEM_B",) in mems
        # No single received context may carry both memories.
        for ctx in received:
            assert not ("MEM_A" in ctx.memory_records and "MEM_B" in ctx.memory_records)

        await svc.stop()
        await memory.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_tool_context_does_not_cross_requests(self, tmp_path):
        """available_tools / history in RequestContext must not cross requests."""
        from capsule_brain.autolearn.controller import DispatcherResult
        from capsule_brain.autolearn.schema import Action, ExecutiveState
        from capsule_brain.conversation.models import RequestContext
        from capsule_brain.conversation.service import ConversationService
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.runtime.service import ServiceState

        class _StubProvider(LLMProvider):
            name = "fake"

            async def generate(self, request, model_cfg):
                from capsule_brain.llm.models import LLMResult
                return LLMResult(text="ok", model="fake", provider="fake", latency_ms=1.0, usage={})

        bus = LocalEventBus()
        await bus.start()
        memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
        await memory.start()
        gw = LLMGateway(
            cfg={
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}
                },
            },
            providers={"fake": _StubProvider()},
        )
        gw.state = ServiceState.RUNNING
        svc = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gw,
            cfg={"db_path": str(tmp_path / "conv.sqlite")},
        )
        await svc.start()

        received: list[RequestContext] = []
        original = svc._respond_with_tools

        async def _recording(state, *, conversation_id=None, request_context=None):
            received.append(request_context)
            return DispatcherResult(text="ok", action_completed=True)

        svc._respond_with_tools = _recording  # type: ignore[assignment]

        state = ExecutiveState(prompt_features={"text": "q"}, available_tools=[])
        ctx_a = RequestContext(
            request_id="ra",
            conversation_id="cA",
            prompt="qA",
            available_tools=("toolA",),
            history=(Turn(role=TurnRole.USER, text="HISTORY_A"),),
        )
        ctx_b = RequestContext(
            request_id="rb",
            conversation_id="cB",
            prompt="qB",
            available_tools=("toolB",),
            history=(Turn(role=TurnRole.USER, text="HISTORY_B"),),
        )
        await asyncio.gather(
            svc.execute_executive_action(
                action=Action.CALL_TOOL,
                state=state,
                conversation_id="cA",
                request_context=ctx_a,
            ),
            svc.execute_executive_action(
                action=Action.CALL_TOOL,
                state=state,
                conversation_id="cB",
                request_context=ctx_b,
            ),
        )
        svc._respond_with_tools = original  # type: ignore[assignment]

        assert len(received) == 2
        tools = {ctx.available_tools for ctx in received}
        assert ("toolA",) in tools
        assert ("toolB",) in tools
        for ctx in received:
            assert not ("toolA" in ctx.available_tools and "toolB" in ctx.available_tools)

        await svc.stop()
        await memory.stop()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_workflow_context_does_not_cross_requests(self, tmp_path):
        """workflow prompt / conversation_id in RequestContext must not cross."""
        from capsule_brain.autolearn.controller import DispatcherResult
        from capsule_brain.autolearn.schema import Action, ExecutiveState
        from capsule_brain.conversation.models import RequestContext
        from capsule_brain.conversation.service import ConversationService
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.runtime.service import ServiceState

        class _StubProvider(LLMProvider):
            name = "fake"

            async def generate(self, request, model_cfg):
                from capsule_brain.llm.models import LLMResult
                return LLMResult(text="ok", model="fake", provider="fake", latency_ms=1.0, usage={})

        bus = LocalEventBus()
        await bus.start()
        memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
        await memory.start()
        gw = LLMGateway(
            cfg={
                "enable": True,
                "default_model": "fake",
                "models": {
                    "fake": {"provider": "fake", "model_name": "fake", "capabilities": ["text"]}
                },
            },
            providers={"fake": _StubProvider()},
        )
        gw.state = ServiceState.RUNNING
        svc = ConversationService(
            event_bus=bus,
            memory=memory,
            llm=gw,
            cfg={"db_path": str(tmp_path / "conv.sqlite")},
        )
        await svc.start()

        received: list[RequestContext] = []
        original = svc._respond_with_workflow

        async def _recording(state, *, conversation_id=None, request_context=None):
            received.append(request_context)
            return DispatcherResult(text="ok", action_completed=True)

        svc._respond_with_workflow = _recording  # type: ignore[assignment]

        state = ExecutiveState(prompt_features={"text": "q"}, available_tools=[])
        ctx_a = RequestContext(request_id="ra", conversation_id="cA", prompt="WORKFLOW_A")
        ctx_b = RequestContext(request_id="rb", conversation_id="cB", prompt="WORKFLOW_B")
        await asyncio.gather(
            svc.execute_executive_action(
                action=Action.START_WORKFLOW,
                state=state,
                conversation_id="cA",
                request_context=ctx_a,
            ),
            svc.execute_executive_action(
                action=Action.START_WORKFLOW,
                state=state,
                conversation_id="cB",
                request_context=ctx_b,
            ),
        )
        svc._respond_with_workflow = original  # type: ignore[assignment]

        assert len(received) == 2
        prompts = {ctx.prompt for ctx in received}
        cids = {ctx.conversation_id for ctx in received}
        assert prompts == {"WORKFLOW_A", "WORKFLOW_B"}
        assert cids == {"cA", "cB"}

        await svc.stop()
        await memory.stop()
        await bus.stop()
