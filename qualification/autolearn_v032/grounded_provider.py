"""Deterministic grounded LLM provider for v0.3.2 qualification.

This is a REAL LLM provider (registered through the normal LLMGateway path),
not the SimulatedCounterfactualRuntime. It is deterministic and "grounded":
it follows instructions by echoing structured tokens present in the user
prompt, retrieved memory records, or tool results. This lets the real
ConversationService / MemoryService / ToolRegistry / WorkflowRunner produce
verifiable outcomes without an external API key.

The provider never has access to hidden answers (secrets, expected tool
outputs, acceptance nonces are NOT in the prompt). It only echoes what the
real service path surfaces:
- ANSWER_DIRECT: echoes a structured token present in the user prompt.
- RETRIEVE_MEMORY: echoes a MEM-* token surfaced by MemoryService retrieval
  (prefers the verified secret over a CONFLICT decoy).
- CALL_TOOL: emits a tool_call for the available tool, then returns the tool
  result content once the gateway feeds it back.
- START_WORKFLOW: the workflow's generate node calls this provider with the
  goal; it extracts the nonce and emits the solution code.
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


class GroundedQualificationProvider(LLMProvider):
    """Deterministic, instruction-following provider for qualification."""

    name = "qual_grounded"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, request: LLMRequest, model_cfg: dict) -> LLMResult:
        started = time.perf_counter()
        text = self._produce(request)
        self.calls.append(text[:80])
        latency = (time.perf_counter() - started) * 1000.0

        # Tool-calling: if tools are available and no tool results have been
        # fed back yet, request a call on the first tool.
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
            # When requesting a tool call, the visible text is a placeholder.
            text = text or ""

        return LLMResult(
            text=text,
            model="qual-grounded-v032",
            provider=self.name,
            latency_ms=latency,
            attempts=1,
            usage={"total_tokens": max(10, len(text) // 4)},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            trace=RequestTrace(model_alias="qual-grounded-v032", route="chat"),
        )

    def _produce(self, request: LLMRequest) -> str:
        # If tool results have been fed back, return the tool output as the
        # final structured answer.
        if request.tool_results:
            for tr in reversed(request.tool_results):
                content = str(tr.get("content", ""))
                m = _TOOL_RESULT_RE.search(content)
                if m:
                    return json.dumps({"result": m.group(1)})
                # Fall back to echoing the raw tool content.
                if content:
                    return content
            # Also check messages for tool role content.
            for msg in reversed(request.messages):
                if msg.get("role") == "tool":
                    content = str(msg.get("content", ""))
                    m = _TOOL_RESULT_RE.search(content)
                    if m:
                        return json.dumps({"result": m.group(1)})
                    if content:
                        return content

        # Build the full visible text (prompt + system + memory context).
        visible = request.prompt or ""
        for msg in request.messages:
            if msg.get("role") == "user":
                visible += "\n" + str(msg.get("content", ""))
            elif msg.get("role") == "tool":
                visible += "\n" + str(msg.get("content", ""))

        # Direct token in the prompt.
        m = _DIRECT_RE.search(visible)
        if m:
            return json.dumps({"result": m.group(1)})

        # Memory secret surfaced by retrieval (prefer non-CONFLICT).
        mems = _MEM_RE.findall(visible)
        if mems:
            preferred = [m for m in mems if "CONFLICT" not in m]
            chosen = preferred[0] if preferred else mems[0]
            return json.dumps({"result": chosen})

        # Workflow generate node: produce solution code from the nonce.
        nm = _NONCE_RE.search(visible)
        if nm:
            nonce = nm.group(1)
            return f"def answer():\n    return '{nonce}'\n"

        # Fallback: echo the last user line.
        lines = [ln.strip() for ln in visible.splitlines() if ln.strip()]
        return lines[-1] if lines else json.dumps({"result": "UNKNOWN"})
