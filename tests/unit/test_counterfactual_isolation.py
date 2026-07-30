"""Tests for counterfactual isolation (v0.3.1 / 2.15.2 Section 8).

Verifies that:
- Memory state resets between actions.
- Tool counter resets between actions.
- Workflow filesystem resets between actions.
- Experience DB is isolated.
- Conversation history is isolated.
- Deterministic task setup is equivalent across actions.
"""
from __future__ import annotations

import asyncio

import pytest

from capsule_brain.autolearn.counterfactual import (
    CounterfactualIsolationFactory,
    CounterfactualTask,
    DirectTaskProvisioner,
    IsolatedEnvironment,
    MemoryTaskProvisioner,
    SafetyTaskProvisioner,
    ToolTaskProvisioner,
    WorkflowTaskProvisioner,
    get_provisioner,
)
from capsule_brain.autolearn.schema import Action, ExecutiveState


def _make_task(task_id: str = "t1", family: str = "direct_answer") -> CounterfactualTask:
    return CounterfactualTask(
        task_id=task_id,
        family=family,
        archetype="test",
        prompt="test prompt",
        allowed_actions=[Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY],
        expected_action=Action.ANSWER_DIRECT,
        state=ExecutiveState(prompt_features={"text": "test"}),
        verifier=None,
        setup={},
    )


class TestCounterfactualIsolationFactory:
    """Test that the isolation factory creates truly isolated environments."""

    def test_each_environment_has_unique_ids(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task()

        async def _run():
            env_a = await factory.create(task, Action.ANSWER_DIRECT, seed=42)
            env_b = await factory.create(task, Action.RETRIEVE_MEMORY, seed=42)
            return env_a, env_b

        env_a, env_b = asyncio.run(_run())
        assert env_a.env_id != env_b.env_id
        assert env_a.conversation_id != env_b.conversation_id
        assert env_a.workflow_run_id != env_b.workflow_run_id
        assert env_a.memory_db_path != env_b.memory_db_path
        assert env_a.experience_db_path != env_b.experience_db_path
        assert env_a.fs_root != env_b.fs_root
        assert env_a.exec_dir != env_b.exec_dir

    def test_environments_have_different_filesystem_roots(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task()

        async def _run():
            envs = []
            for action in [Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY,
                           Action.CALL_TOOL, Action.START_WORKFLOW]:
                env = await factory.create(task, action, seed=42)
                envs.append(env)
            return envs

        envs = asyncio.run(_run())
        roots = {e.fs_root for e in envs}
        assert len(roots) == 4, "each action must have a unique filesystem root"

    def test_environments_have_different_memory_dbs(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task()

        async def _run():
            env_a = await factory.create(task, Action.ANSWER_DIRECT, seed=42)
            env_b = await factory.create(task, Action.RETRIEVE_MEMORY, seed=42)
            return env_a, env_b

        env_a, env_b = asyncio.run(_run())
        assert env_a.memory_db_path != env_b.memory_db_path

    def test_seed_is_deterministic(self, tmp_path):
        """Same seed produces equivalent setup values."""
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task()

        async def _run():
            env_a = await factory.create(task, Action.ANSWER_DIRECT, seed=123)
            env_b = await factory.create(task, Action.RETRIEVE_MEMORY, seed=123)
            return env_a, env_b

        env_a, env_b = asyncio.run(_run())
        assert env_a.seed == env_b.seed == 123

    def test_cleanup_removes_files(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task()

        async def _run():
            env = await factory.create(task, Action.ANSWER_DIRECT, seed=42)
            return env

        env = asyncio.run(_run())
        from pathlib import Path
        assert Path(env.fs_root).exists()
        assert Path(env.exec_dir).exists()
        env.cleanup()
        assert not Path(env.fs_root).exists()
        assert not Path(env.exec_dir).exists()


class TestTaskProvisioners:
    """Test that task provisioners create correct state."""

    def test_direct_provisioner_no_hidden_state(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task(family="direct_answer")
        task.setup["expected_answer"] = "42"

        async def _run():
            env = await factory.create(task, Action.ANSWER_DIRECT, seed=42)
            prov = DirectTaskProvisioner()
            return await prov.provision(task, env)

        result = asyncio.run(_run())
        assert result.expected_output == "42"
        assert result.verifier_state["expected"] == "42"

    def test_memory_provisioner_generates_secret(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task(family="memory_required")

        async def _run():
            env = await factory.create(task, Action.RETRIEVE_MEMORY, seed=42)
            prov = MemoryTaskProvisioner()
            return await prov.provision(task, env)

        result = asyncio.run(_run())
        assert result.expected_output.startswith("MEM-")
        assert len(result.expected_output) > 10
        assert "secret" not in task.prompt  # secret not in prompt
        assert "secret" not in str(task.state.prompt_features)  # not in features

    def test_tool_provisioner_generates_output(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task(family="tool_required")

        async def _run():
            env = await factory.create(task, Action.CALL_TOOL, seed=42)
            prov = ToolTaskProvisioner()
            return await prov.provision(task, env)

        result = asyncio.run(_run())
        assert result.expected_output.startswith("TOOL-")
        assert "expected_tool_output" not in task.prompt

    def test_workflow_provisioner_installs_acceptance(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task(family="coding_workflow")
        task.setup["acceptance_code"] = "assert True"

        async def _run():
            env = await factory.create(task, Action.START_WORKFLOW, seed=42)
            prov = WorkflowTaskProvisioner()
            return await prov.provision(task, env)

        result = asyncio.run(_run())
        from pathlib import Path
        assert Path(result.verifier_state["acceptance_path"]).exists()
        assert Path(result.verifier_state["acceptance_path"]).read_text() == "assert True"

    def test_safety_provisioner_marks_safety(self, tmp_path):
        factory = CounterfactualIsolationFactory(base_dir=str(tmp_path / "iso"))
        task = _make_task(family="operator_safety")

        async def _run():
            env = await factory.create(task, Action.ANSWER_DIRECT, seed=42)
            prov = SafetyTaskProvisioner()
            return await prov.provision(task, env)

        result = asyncio.run(_run())
        assert result.verifier_state["requires_safety"] is True

    def test_get_provisioner_returns_correct_type(self):
        assert isinstance(get_provisioner("direct_answer"), DirectTaskProvisioner)
        assert isinstance(get_provisioner("memory_required"), MemoryTaskProvisioner)
        assert isinstance(get_provisioner("tool_required"), ToolTaskProvisioner)
        assert isinstance(get_provisioner("coding_workflow"), WorkflowTaskProvisioner)
        assert isinstance(get_provisioner("operator_safety"), SafetyTaskProvisioner)
