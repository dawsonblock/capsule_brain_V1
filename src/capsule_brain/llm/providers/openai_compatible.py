from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from capsule_brain.llm.errors import LLMConfigurationError, LLMProviderError
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.tools import ToolCall
from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        # Disable httpx's default 5s read timeout. Request timeout is managed
        # at the gateway level via asyncio.wait_for; without this override,
        # longer generations silently abort with httpx.ReadTimeout.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def generate(self, request: LLMRequest, model_cfg: dict[str, Any]) -> LLMResult:
        api_base = str(model_cfg.get("api_base", "https://api.openai.com/v1")).rstrip("/")
        model_name = str(model_cfg.get("model_name", "")).strip()
        key_env = str(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(key_env)

        if not model_name:
            raise LLMConfigurationError("model_name is required")
        if not api_key:
            raise LLMConfigurationError(f"Missing API key environment variable: {key_env}")

        messages = self._build_messages(request)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None
                else model_cfg.get("temperature", 0.3),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]
            # Respect an explicit caller policy; otherwise let the model choose.
            payload["tool_choice"] = request.metadata.get("tool_choice", "auto")

        started = time.perf_counter()
        response = await self.client.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.is_error:
            raise LLMProviderError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Malformed provider response") from exc

        tool_calls = self._parse_tool_calls(message)

        usage = data.get("usage", {})
        return LLMResult(
            text=text or "",
            model=model_name,
            provider=self.name,
            latency_ms=latency_ms,
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
            raw=data,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _build_messages(request: LLMRequest) -> list[dict[str, Any]]:
        """Build a protocol-valid OpenAI chat transcript.

        ``request.messages`` is authoritative when present. This preserves the
        assistant message containing ``tool_calls`` immediately before the
        matching ``tool`` role messages. The legacy prompt/system/tool_results
        path remains for callers that do not use the canonical transcript.
        """
        if request.messages:
            return [dict(message) for message in request.messages]

        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        for result in request.tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id", ""),
                    "content": str(result.get("content", "")),
                }
            )
        return messages

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Parse provider tool calls without hiding malformed arguments."""
        raw_calls = message.get("tool_calls") or []
        calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            call_id = str(raw.get("id", ""))
            function = raw.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            args_blob = function.get("arguments", "{}")
            if isinstance(args_blob, dict):
                calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=dict(args_blob),
                        raw_arguments=json.dumps(args_blob),
                    )
                )
                continue
            raw_arguments = str(args_blob or "{}")
            try:
                decoded = json.loads(raw_arguments)
                if not isinstance(decoded, dict):
                    raise ValueError("tool arguments must decode to a JSON object")
                calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=decoded,
                        raw_arguments=raw_arguments,
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments={},
                        raw_arguments=raw_arguments,
                        parse_error=str(exc),
                    )
                )
        return calls

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
