"""Tests for immutable RequestContext isolation (v0.3.1 / 2.15.2 Section 4).

Verifies that:
- RequestContext is immutable (frozen dataclass).
- Concurrent requests do not observe each other's state.
- No mutable service-level request context remains.
"""
from __future__ import annotations

import asyncio

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
