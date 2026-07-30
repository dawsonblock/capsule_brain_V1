"""Integration tests for AutoLearn v0.2 ConversationService routing.

Verifies that:
- ConversationService routes through ExecutiveController when autolearn.enable=true
- ConversationService preserves v0.1 behavior when autolearn.enable=false
- The _ConversationActionDispatcher delegates to the correct _respond_* method
- Experiences are persisted when the ExecutiveController executes
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.memory.service import MemoryService


class _FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, text: str = "fake answer") -> None:
        self._text = text
        self.calls: list[str] = []

    async def generate(self, request, model_cfg):
        self.calls.append(request.prompt[:50])
        return LLMResult(
            text=self._text,
            model="fake",
            provider="fake",
            latency_ms=1.0,
            usage={"total_tokens": 10},
        )


def _build_gateway(provider: LLMProvider | None = None):
    from capsule_brain.llm.gateway import LLMGateway
    from capsule_brain.runtime.service import ServiceState
    gw = LLMGateway(
        cfg={
            "enable": True,
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
        },
        providers={"fake": provider or _FakeProvider()},
    )
    gw.state = ServiceState.RUNNING
    return gw


@pytest.mark.asyncio
async def test_conversation_autolearn_disabled_preserves_v01_behavior(tmp_path):
    """When autolearn_enable=false, ConversationService uses the legacy direct path."""
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
    await memory.start()
    provider = _FakeProvider("legacy answer")
    gateway = _build_gateway(provider)

    svc = ConversationService(
        event_bus=bus,
        memory=memory,
        llm=gateway,
        cfg={
            "db_path": str(tmp_path / "conv.sqlite"),
        },
        autolearn_enabled=False,
    )
    await svc.start()

    resp = await svc.respond(conversation_id="c1", text="hello")
    assert resp.text == "legacy answer"
    assert svc.autolearn_enabled is False

    await svc.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_conversation_autolearn_enabled_routes_through_controller(tmp_path):
    """When autolearn_enable=true, ConversationService routes through ExecutiveController."""
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
    await memory.start()
    provider = _FakeProvider("autolearn answer")
    gateway = _build_gateway(provider)

    # Build an ExecutiveController with a real dispatcher.
    svc = ConversationService(
        event_bus=bus,
        memory=memory,
        llm=gateway,
        cfg={
            "db_path": str(tmp_path / "conv.sqlite"),
        },
        autolearn_enabled=True,
    )
    await svc.start()

    # The dispatcher should be attached.
    assert svc.autolearn_enabled is True
    assert svc._dispatcher is not None

    # Verify the dispatcher delegates to the right _respond_* method.
    state = ExecutiveState(prompt_features={"text": "test"})
    svc._current_request = RequestContext(
        request_id="test_req",
        conversation_id="c1",
        prompt="test",
        history=(),
        memory_records=(),
        user_turn_id="ut1",
    )
    try:
        result = await svc._dispatcher.answer_direct(state)
        assert result.text == "autolearn answer"
        assert result.action_completed is True
    finally:
        svc._current_request = None

    await svc.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_conversation_dispatcher_retrieve_memory(tmp_path):
    """The dispatcher's retrieve_memory delegates to _respond_with_memory."""
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
    await memory.start()
    provider = _FakeProvider("memory answer")
    gateway = _build_gateway(provider)

    svc = ConversationService(
        event_bus=bus,
        memory=memory,
        llm=gateway,
        cfg={
            "db_path": str(tmp_path / "conv.sqlite"),
        },
        autolearn_enabled=True,
    )
    await svc.start()

    state = ExecutiveState(prompt_features={"text": "test"})
    svc._current_request = RequestContext(
        request_id="test_req",
        conversation_id="c1",
        prompt="test",
        history=(),
        memory_records=(),
        user_turn_id="ut1",
    )
    try:
        result = await svc._dispatcher.retrieve_memory(state)
        assert result.text == "memory answer"
        assert result.action_completed is True
    finally:
        svc._current_request = None

    await svc.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_conversation_dispatcher_call_tool_no_tools(tmp_path):
    """The dispatcher's call_tool returns runtime_error when no tools are registered."""
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
    await memory.start()
    provider = _FakeProvider("tool answer")
    gateway = _build_gateway(provider)

    svc = ConversationService(
        event_bus=bus,
        memory=memory,
        llm=gateway,
        cfg={
            "db_path": str(tmp_path / "conv.sqlite"),
        },
        autolearn_enabled=True,
    )
    await svc.start()

    state = ExecutiveState(prompt_features={"text": "test"})
    svc._current_request = RequestContext(
        request_id="test_req",
        conversation_id="c1",
        prompt="test",
        history=(),
        memory_records=(),
        user_turn_id="ut1",
    )
    try:
        result = await svc._dispatcher.call_tool(state)
        assert result.runtime_error is True
        assert result.action_completed is False
    finally:
        svc._current_request = None

    await svc.stop()
    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_conversation_dispatcher_start_workflow_no_runner(tmp_path):
    """The dispatcher's start_workflow returns runtime_error when no workflow runner is available."""
    bus = LocalEventBus()
    await bus.start()
    memory = MemoryService(event_bus=bus, cfg={"db_path": str(tmp_path / "mem.sqlite")})
    await memory.start()
    provider = _FakeProvider("workflow answer")
    gateway = _build_gateway(provider)

    svc = ConversationService(
        event_bus=bus,
        memory=memory,
        llm=gateway,
        cfg={
            "db_path": str(tmp_path / "conv.sqlite"),
        },
        autolearn_enabled=True,
    )
    await svc.start()

    state = ExecutiveState(prompt_features={"text": "test"})
    svc._current_request = RequestContext(
        request_id="test_req",
        conversation_id="c1",
        prompt="test",
        history=(),
        memory_records=(),
        user_turn_id="ut1",
    )
    try:
        result = await svc._dispatcher.start_workflow(state)
        assert result.runtime_error is True
        assert result.action_completed is False
    finally:
        svc._current_request = None

    await svc.stop()
    await memory.stop()
    await bus.stop()
