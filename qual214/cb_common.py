"""Shared configuration and helpers for the Capsule Brain v2.14.0
local-LLM qualification harness.

All tests drive the REAL installed capsule_brain 2.14.0 runtime
(build_application / ConversationService / ToolRegistry / WorkflowRunnerService
/ ExecutionService / VerificationService). No runtime class is monkeypatched
or replaced with manual subprocess logic.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ollama exposes an OpenAI-compatible endpoint. The OpenAI-compatible
# provider requires an api key env var; for local Ollama any non-empty
# dummy value works.
os.environ.setdefault("OLLAMA_API_KEY", "ollama")

ROOT = Path(__file__).resolve().parent
STORES = ROOT / "stores"
STORES.mkdir(parents=True, exist_ok=True)

# Primary recommended model.
QWEN_CODER = "qwen2.5-coder:3b"
# Secondary model used ONLY where the recommended model lacks a required
# capability (native tool-call emission). Confirmed at the raw endpoint.
LLAMA31 = "llama3.1:8b"

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")


def base_cfg(workdir: Path, *, model: str = QWEN_CODER) -> dict[str, Any]:
    """A fully-connected config: llm + conversation + tools + execution +
    verification + workflow, acceptance verification mode, smoke success
    disabled."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = workdir / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return {
        "memory": {"db_path": str(workdir / "memory.sqlite")},
        "memory_consolidator": {"enable": False},
        "llm_gateway": {
            "enable": True,
            "default_model": "primary",
            "timeout_s": 90.0,
            "max_attempts": 1,
            "models": {
                "primary": {
                    "provider": "openai",
                    "model_name": model,
                    "api_base": OLLAMA_BASE,
                    "api_key_env": "OLLAMA_API_KEY",
                    "capabilities": ["text", "json", "tools"],
                    "input_cost_per_million": 0.0,
                    "output_cost_per_million": 0.0,
                },
            },
            "routing": {
                "routes": {
                    "chat": {"models": ["primary"]},
                    "coding": {"models": ["primary"]},
                    "reflection": {"models": ["primary"]},
                }
            },
        },
        "tools": {"allow_side_effects": True},
        "conversation": {
            "enable": True,
            "db_path": str(workdir / "conversation.sqlite"),
            "use_tools": True,
            "route": "chat",
            "history_limit": 12,
            "memory_limit": 8,
        },
        "feedback": {"enable": False},
        "reflection": {
            "enable": True,
            "db_path": str(workdir / "reflection.sqlite"),
            "max_iterations": 3,
        },
        "execution": {
            "enable": True,
            "runner": "host",
            "unsafe_allow_host_execution": True,
            "allow": True,
            "cwd_root": str(sandbox),
            "allowed_commands": ["python", "python3", "pytest"],
            "timeout_s": 30,
            "max_output_chars": 40000,
            "worker_pool": {"max_workers": 0},
        },
        "verification": {
            "enable": True,
            "db_path": str(workdir / "verification.sqlite"),
        },
        "workflow": {
            "enable": True,
            "db_path": str(workdir / "workflows.sqlite"),
            "verification_mode": "acceptance",
            "allow_smoke_success": False,
            "max_reflect_iterations": 3,
            "auto_resume": False,
        },
        "goal_planner": {"db_path": str(workdir / "goals.sqlite")},
        "learning": {"db_path": str(workdir / "experience.sqlite")},
        "redis_bridge": {"enable": False},
        "tracing": {"enable": False},
    }


def new_workdir(label: str) -> Path:
    return STORES / f"{label}_{uuid.uuid4().hex[:8]}"


def assert_services_present(app) -> list[str]:
    """Test 2 helper: assert the full application graph is present."""
    required = [
        "llm_gateway",
        "tool_registry",
        "conversation",
        "experience_store",
        "execution",
        "verification",
        "workflow_runner",
    ]
    missing = [n for n in required if app.services.get(n) is None]
    if missing:
        raise AssertionError(f"Application graph incomplete; missing: {missing}")
    return required
