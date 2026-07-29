"""v2.14.0 connected-build regression tests.

These three tests pin the corrections that distinguish the connected v2.14.0
build from v2.13.1:

1. ``test_v214_bootstrap_rejects_enabled_typo`` — the construction-boundary
   config validator rejects ``<section>.enabled`` and demands ``enable``.
   v2.13.1 silently ignored the typo, which previously let a qualification
   run appear to boot correctly while major services were absent.

2. ``test_v214_normalize_python_artifact_strips_outer_fence`` — the
   ``normalize_python_artifact`` helper conservatively removes a single
   outer Markdown fence from a generated Python artifact (a transport
   defect common with small local chat models) while leaving nested fences
   or prose untouched so verification still fails closed.

3. ``test_v214_reflection_prompt_is_verifier_conditioned`` — the reflect
   node's repair prompt feeds the verification kind and the verifier /
   acceptance failure text into the repair request and instructs the model
   not to weaken or bypass the acceptance test. v2.13.1 used a much weaker
   generic "produce corrected code" prompt.

No production source is monkeypatched. The tests exercise the real
``build_application`` and the real built-in workflow builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.execution.models import ExecutionPolicy
from capsule_brain.execution.runner import ExecutionRunner
from capsule_brain.execution.service import ExecutionService
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import ConfigurationError, build_application
from capsule_brain.workflow.builtins import (
    build_plan_generate_test_reflect_workflow,
    normalize_python_artifact,
)
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.workflow.repository import WorkflowRepository
from capsule_brain.workflow.runner import WorkflowRunnerService


# ---------------------------------------------------------------------------
# Correction 1: configuration typo rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("section", [
    "llm_gateway", "conversation", "feedback", "reflection", "execution",
    "verification", "workflow", "redis_bridge", "memory_consolidator",
])
def test_v214_bootstrap_rejects_enabled_typo(tmp_path, section):
    """``<section>.enabled`` must be rejected in favor of ``<section>.enable``."""
    cfg = {
        "memory": {"db_path": str(tmp_path / "memory.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {
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
        "conversation": {"enable": True, "db_path": str(tmp_path / "conv.sqlite")},
        "execution": {"enable": False},
        "workflow": {"enable": False},
    }
    # Inject the offending ``enabled`` typo on the target section. For sections
    # that already have a config dict, merge it; otherwise create one.
    base = dict(cfg.get(section, {}))
    base["enabled"] = True
    cfg[section] = base
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        build_application(cfg)


def test_v214_bootstrap_accepts_enable_keyword(tmp_path):
    """The correct ``enable`` keyword must still boot cleanly."""
    cfg = {
        "memory": {"db_path": str(tmp_path / "memory.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {
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
        "conversation": {"enable": True, "db_path": str(tmp_path / "conv.sqlite")},
        "execution": {"enable": False},
        "workflow": {"enable": False},
    }

    class _FakeProvider(LLMProvider):
        name = "fake"

        async def generate(self, request, model_cfg):
            return LLMResult(text="ok", model="fake", provider="fake", latency_ms=1)

        async def close(self):
            pass

    app = build_application(cfg, llm_providers={"fake": _FakeProvider()})
    assert app.services.get("conversation") is not None
    assert app.services.get("llm_gateway") is not None


# ---------------------------------------------------------------------------
# Correction 2: Python artifact normalization
# ---------------------------------------------------------------------------

def test_v214_normalize_python_artifact_strips_outer_fence():
    raw = "```python\ndef add(a, b):\n    return a + b\n```"
    normalized, changed = normalize_python_artifact(raw)
    assert changed is True
    assert normalized == "def add(a, b):\n    return a + b"


def test_v214_normalize_python_artifact_strips_bare_fence():
    raw = "```\nprint('hi')\n```"
    normalized, changed = normalize_python_artifact(raw)
    assert changed is True
    assert normalized == "print('hi')"


def test_v214_normalize_python_artifact_leaves_plain_code_untouched():
    raw = "def add(a, b):\n    return a + b"
    normalized, changed = normalize_python_artifact(raw)
    assert changed is False
    assert normalized == raw


def test_v214_normalize_python_artifact_refuses_nested_fences():
    # An inner fence means this is not a simple outer wrapper; leave it so
    # verification fails closed rather than silently stripping structure.
    raw = "```python\nx = '```'\nprint(x)\n```"
    normalized, changed = normalize_python_artifact(raw)
    assert changed is False
    assert normalized == raw.strip()


def test_v214_normalize_python_artifact_refuses_prose_trailer():
    # A non-fence trailing line means this is prose-wrapped, not a clean
    # artifact; do not normalize.
    raw = "```python\nprint('hi')\n```\nHere is the explanation."
    normalized, changed = normalize_python_artifact(raw)
    assert changed is False


def test_v214_normalize_python_artifact_empty():
    assert normalize_python_artifact("") == ("", False)
    assert normalize_python_artifact("   ") == ("", False)


# ---------------------------------------------------------------------------
# Correction 3: verifier-conditioned reflection prompt
# ---------------------------------------------------------------------------

class _RecordingProvider(LLMProvider):
    """Records the last prompt it received so the test can assert the
    verifier-conditioned repair instructions are present."""
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.last_prompt = ""

    async def generate(self, request, model_cfg):
        self.calls += 1
        self.last_prompt = request.prompt
        if self.calls == 1:
            text = "1. Write the function"  # plan
        elif self.calls == 2:
            # First generate: broken code (wrong operator) so acceptance fails.
            text = "def add(a, b):\n    return a - b\n"
        else:
            # Reflect: repaired code.
            text = "def add(a, b):\n    return a + b\n"
        return LLMResult(text=text, model="fake", provider="fake", latency_ms=1)

    async def close(self):
        pass


def _acceptance_gateway(provider, tmp_path):
    return LLMGateway(
        {
            "default_model": "fake",
            "models": {
                "fake": {
                    "provider": "fake",
                    "model_name": "fake",
                    "capabilities": ["text"],
                }
            },
            "routing": {
                "routes": {
                    "coding": {"models": ["fake"]},
                    "reflection": {"models": ["fake"]},
                }
            },
        },
        providers={"fake": provider},
    )


@pytest.mark.asyncio
async def test_v214_reflection_prompt_is_verifier_conditioned(tmp_path):
    """The reflect node must feed the verification kind and the verifier /
    acceptance failure text into the repair prompt and instruct the model
    not to weaken or bypass the acceptance test."""
    bus = LocalEventBus()
    await bus.start()
    root = tmp_path / "sandbox"
    root.mkdir()
    executable = Path(sys.executable).name
    execution = ExecutionService(
        bus,
        cfg={
            "allow": True,
            "cwd_root": str(root),
            "allowed_commands": [executable],
            "timeout_s": 5,
        },
        runner=ExecutionRunner(
            ExecutionPolicy(
                allow=True,
                cwd_root=str(root),
                allowed_commands=(executable,),
                timeout_s=5,
            )
        ),
    )
    await execution.start()

    provider = _RecordingProvider()
    gateway = _acceptance_gateway(provider, tmp_path)
    await gateway.start()

    runner = WorkflowRunnerService(
        bus, repository=WorkflowRepository(str(tmp_path / "wf.sqlite"))
    )
    runner.register_workflow(
        build_plan_generate_test_reflect_workflow(
            services={"llm_gateway": gateway, "execution": execution},
            max_iterations=2,
        )
    )
    await runner.start()

    acceptance = (
        "from solution import add\n"
        "assert add(2, 3) == 5\n"
        "assert add(-1, 1) == 0\n"
        "print('ACCEPTANCE_OK')\n"
    )
    run = await runner.start_run(
        workflow_name="plan_generate_test_reflect",
        goal="implement add(a, b) returning a + b",
        metadata={
            "verification": {
                "files": {"test_solution.py": acceptance},
                "command": [executable, "test_solution.py"],
            }
        },
    )

    try:
        # The run should repair and complete.
        assert run.status == WorkflowStatus.COMPLETED
        assert run.state.extra["test_passed"] is True
        assert run.state.iterations >= 1

        # The reflect prompt (provider call >= 3) must be verifier-conditioned.
        assert provider.calls >= 3
        prompt = provider.last_prompt
        assert "VERIFICATION KIND" in prompt
        assert "VERIFIER / ACCEPTANCE FAILURE" in prompt
        assert "REPAIR REQUIREMENTS" in prompt
        # The actual acceptance failure (AssertionError) must reach the prompt.
        assert "AssertionError" in prompt or "assert" in prompt.lower()
        # The model must be told not to weaken/bypass the acceptance test.
        assert "Do not weaken or bypass the acceptance test" in prompt
        assert "Return raw Python source only" in prompt
    finally:
        await runner.stop()
        await gateway.stop()
        await execution.stop()
        await bus.stop()
