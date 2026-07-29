from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.runtime.service import CapsuleService


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Declarative description of a tool the model may call.

    Mirrors the OpenAI / Anthropic function-calling schema so it can be
    serialized directly into a chat-completions request:
    ``{"type": "function", "function": {"name", "description", "parameters"}}``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    timeout_s: float | None = None
    max_result_chars: int | None = None
    permission: str = "standard"
    side_effecting: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True, frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    parse_error: str | None = None


@dataclass(slots=True, frozen=True)
class ToolResult:
    """The outcome of executing a tool call.

    ``content`` is the stringified payload returned to the model in the
    ``tool`` role message. ``is_error`` lets the model distinguish a tool
    failure from a successful return.
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class ToolRegistry(CapsuleService):
    """A managed registry of callable tools exposed to the LLM gateway.

    Tools are registered with a JSON Schema describing their parameters and
    an async handler that receives the validated arguments. The registry
    validates incoming tool calls against the schema before dispatching, so
    handlers can assume well-typed arguments.

    The registry is a CapsuleService so it participates in the normal
    dependency-ordered startup/shutdown lifecycle alongside the gateway.
    """

    name = "tool_registry"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self.default_timeout_s = max(0.1, float(self.cfg.get("timeout_s", 15.0)))
        self.default_max_result_chars = max(256, int(self.cfg.get("max_result_chars", 20000)))
        self.allowed_permissions = {
            str(v) for v in self.cfg.get("allowed_permissions", ["standard"])
        }
        self.allow_side_effects = bool(self.cfg.get("allow_side_effects", False))
        self._calls = 0
        self._failures = 0

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
    ) -> None:
        if not spec.name:
            raise ValueError("ToolSpec.name must be non-empty")
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        try:
            from jsonschema import Draft202012Validator
            Draft202012Validator.check_schema(spec.parameters)
        except ImportError as exc:
            raise RuntimeError(
                "ToolRegistry requires jsonschema for argument validation"
            ) from exc
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)
        self._handlers.pop(name, None)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def to_openai_schema(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def has(self, name: str) -> bool:
        return name in self._specs

    async def execute(self, call: ToolCall) -> ToolResult:
        self._calls += 1
        spec = self._specs.get(call.name)
        if spec is None:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        if call.parse_error:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Invalid JSON tool arguments: {call.parse_error}",
                is_error=True,
                metadata={"raw_arguments": call.raw_arguments or ""},
            )
        if spec.permission not in self.allowed_permissions:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool permission denied: {spec.permission}",
                is_error=True,
            )
        if spec.side_effecting and not self.allow_side_effects:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="Side-effecting tools are disabled by policy",
                is_error=True,
            )

        missing = self._missing_required_keys(spec, call.arguments)
        if missing:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    "Missing required arguments: "
                    + ", ".join(sorted(missing))
                ),
                is_error=True,
            )

        try:
            from jsonschema import Draft202012Validator
            errors = sorted(
                Draft202012Validator(spec.parameters).iter_errors(call.arguments),
                key=lambda err: list(err.path),
            )
        except Exception as exc:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool schema validation failed internally: {exc}",
                is_error=True,
            )
        if errors:
            self._failures += 1
            detail = "; ".join(error.message for error in errors[:5])
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Invalid tool arguments: {detail}",
                is_error=True,
            )

        handler = self._handlers[call.name]
        timeout_s = spec.timeout_s or self.default_timeout_s
        try:
            output = await asyncio.wait_for(handler(call.arguments), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool {call.name} timed out after {timeout_s:.1f}s",
                is_error=True,
            )
        except Exception as exc:
            self._failures += 1
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool {call.name} raised: {exc}",
                is_error=True,
            )

        content = self._stringify(output)
        max_chars = spec.max_result_chars or self.default_max_result_chars
        metadata: dict[str, Any] = {}
        if len(content) > max_chars:
            content = content[:max_chars] + "...[truncated]"
            metadata["truncated"] = True
            metadata["max_result_chars"] = max_chars
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _missing_required_keys(
        spec: ToolSpec, arguments: dict[str, Any]
    ) -> set[str]:
        required = spec.parameters.get("required") or []
        return {str(r) for r in required if r not in arguments}

    @staticmethod
    def _stringify(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output)
        except (TypeError, ValueError):
            return str(output)

    async def health(self):
        from capsule_brain.runtime.service import HealthStatus

        return HealthStatus(
            state=self.state,
            details={
                "tools": sorted(self._specs),
                "tool_count": len(self._specs),
                "calls": self._calls,
                "failures": self._failures,
                "allowed_permissions": sorted(self.allowed_permissions),
                "allow_side_effects": self.allow_side_effects,
            },
        )
