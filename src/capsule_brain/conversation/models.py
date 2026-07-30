from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable per-request context (v0.3.1 / 2.15.2 Section 4).

    Replaces the mutable ``self._current_request`` dict on
    ConversationService. Each concurrent request gets its own immutable
    context that cannot be overwritten by another request.

    The context is passed explicitly through the action dispatch path
    so that no shared mutable state is used.
    """

    request_id: str
    conversation_id: str
    prompt: str
    history: tuple[Any, ...] = ()
    memory_records: tuple[Any, ...] = ()
    available_tools: tuple[str, ...] = ()
    correlation_id: str | None = None
    user_turn_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Conversation:
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Turn:
    id: str = field(default_factory=lambda: uuid4().hex)
    conversation_id: str = ""
    role: TurnRole = TurnRole.USER
    text: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    parent_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AssistantResponse:
    id: str = field(default_factory=lambda: uuid4().hex)
    conversation_id: str = ""
    parent_turn_id: str = ""
    turn_id: str = ""
    text: str = ""
    model_alias: str | None = None
    provider: str | None = None
    model_name: str | None = None
    latency_ms: float | None = None
    attempts: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    route: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
