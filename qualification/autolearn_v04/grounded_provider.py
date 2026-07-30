"""Deterministic grounded LLM provider for v0.4.0 qualification.

Section 11.2: This is the RUNTIME INTEGRATION PROVIDER (formerly
"qual_grounded"). It is a REAL LLM provider registered through the normal
LLMGateway path, but it is deterministic and "grounded" -- it follows
instructions by echoing structured tokens present in the user prompt,
retrieved memory records, or tool results.

It may ONLY qualify:
  * infrastructure;
  * deterministic action execution;
  * verifier contracts;
  * isolation;
  * serialization;
  * reporting.

It MUST NOT be used for the final causal-learning claim (Gate A). Gate A
scientific qualification is BLOCKED under this provider.

The provider never has access to hidden answers (secrets, expected tool
outputs, acceptance nonces are NOT in the prompt). It only echoes what the
real service path surfaces.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.llm.telemetry import RequestTrace
from capsule_brain.llm.tools import ToolCall

_DIRECT_RE = re.compile(r'\{"result":"(DIRECT-[0-9a-f]+)"\}')
_MEM_RE = re.compile(r"\b(MEM-[0-9a-f]+)\b")
_NONCE_RE = re.compile(r"token\s+([0-9a-f]{6,})")
_TOOL_RESULT_RE = re.compile(r'(TOOL-[0-9a-f]+)')

# Provider name kept as "qual_grounded" for LLMGateway registration
# compatibility, but the config labels it PROVIDER_INFRASTRUCTURE.
PROVIDER_NAME = "qual_grounded"


class GroundedQualificationProvider(LLMProvider):
    """Runtime integration provider (infrastructure-only, deterministic).

    This provider is NOT sufficient for a causal learning claim. It exists
    to exercise the real Capsule Brain service stack (ConversationService,
    MemoryService, ToolRegistry, WorkflowRunner, ExecutionService) without
    an external API key, proving the runtime wiring and verifier contracts.
    """

    name = PROVIDER_NAME

    # Marker so callers can detect this is the infrastructure-only provider.
    is_infrastructure_only: bool = True
    # Provider classification (Section 4).
    provider_class = "infrastructure"
    supports_gate_a_claim = False
    supports_gate_b_claim = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def provider_capabilities(self):
        from .provider_classification import classify_grounded_provider
        return classify_grounded_provider()

    async def generate(self, request: LLMRequest, model_cfg: dict) -> LLMResult:
        started = time.perf_counter()
        text = self._produce(request)
        self.calls.append(text[:80])
        latency = (time.perf_counter() - started) * 1000.0
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"
        if request.tools and not request.tool_results:
            spec = request.tools[0]
            tool_calls = [
                ToolCall(
                    id="call_0",
                    name=spec.name,
                    arguments={},
                    raw_arguments="{}",
                )
            ]
            finish_reason = "tool_calls"
            text = text or ""
        return LLMResult(
            text=text,
            model="qual-grounded-v04",
            provider=self.name,
            latency_ms=latency,
            attempts=1,
            usage={"total_tokens": max(10, len(text) // 4)},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            trace=RequestTrace(model_alias="qual-grounded-v04", route="chat"),
        )

    def _produce(self, request: LLMRequest) -> str:
        if request.tool_results:
            for tr in reversed(request.tool_results):
                content = str(tr.get("content", ""))
                m = _TOOL_RESULT_RE.search(content)
                if m:
                    return json.dumps({"result": m.group(1)})
                if content:
                    return content
            for msg in reversed(request.messages):
                if msg.get("role") == "tool":
                    content = str(msg.get("content", ""))
                    m = _TOOL_RESULT_RE.search(content)
                    if m:
                        return json.dumps({"result": m.group(1)})
                    if content:
                        return content
        visible = request.prompt or ""
        for msg in request.messages:
            if msg.get("role") == "user":
                visible += "\n" + str(msg.get("content", ""))
            elif msg.get("role") == "tool":
                visible += "\n" + str(msg.get("content", ""))
        m = _DIRECT_RE.search(visible)
        if m:
            return json.dumps({"result": m.group(1)})
        mems = _MEM_RE.findall(visible)
        if mems:
            preferred = [m for m in mems if "CONFLICT" not in m and "STALE" not in m]
            chosen = preferred[0] if preferred else mems[0]
            return json.dumps({"result": chosen})
        nm = _NONCE_RE.search(visible)
        if nm:
            nonce = nm.group(1)
            return f"def answer():\n    return '{nonce}'\n"
        lines = [ln.strip() for ln in visible.splitlines() if ln.strip()]
        return lines[-1] if lines else json.dumps({"result": "UNKNOWN"})
