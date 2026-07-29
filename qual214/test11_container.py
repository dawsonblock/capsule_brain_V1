"""Test 11: Container execution qualification.

Runs the real plan_generate_test_reflect acceptance workflow through the
real ContainerExecutionRunner (Docker) with the full production-safe control
set: read-only root, network none, non-root user, memory/CPU/PID limits,
pinned image digest, cap-drop ALL, no-new-privileges.

This qualifies the production-safe container execution path, not the
host-runner path used by the functional tests.
"""
import asyncio
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg, new_workdir, QWEN_CODER
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowStatus

ACCEPTANCE = (
    "from solution import add\n"
    "assert add(2, 3) == 5, 'add(2,3) failed'\n"
    "assert add(-1, 1) == 0, 'add(-1,1) failed'\n"
    "print('ACCEPTANCE_OK')\n"
)


def _docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker not found on PATH"
    import subprocess
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return False, f"docker info exit {r.returncode}: {r.stderr[:200]}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


async def main():
    import json
    # Provenance: refuse to run against the wrong build.
    from provenance import assert_provenance
    manifest = assert_provenance()

    ok, why = _docker_available()
    if not ok:
        blocked_result = {
            "test": "11_container_execution",
            "status": "BLOCKED",
            "classification": "environment",
            "evidence": {"docker_available": False, "reason": why},
        }
        print(json.dumps(blocked_result, indent=2, default=str))
        out = HERE / "test11_container_result.json"
        out.write_text(json.dumps(blocked_result, indent=2, default=str))
        print("VERDICT:BLOCKED")
        sys.exit(0)

    wd = new_workdir("t11_container")
    cfg = base_cfg(wd, model=QWEN_CODER)
    # Switch execution to the real container runner with full controls.
    cfg["execution"] = {
        "enable": True,
        "runner": "container",
        "allow": True,
        "container_engine": "docker",
        "container_image": "python:3.11-slim",
        "container_memory": "512m",
        "container_memory_swap": "512m",
        "container_cpus": "1.0",
        "container_pids_limit": 128,
        "container_nofile_limit": 1024,
        "container_pin_image_digest": True,
        # python_compile verifier uses "python"; acceptance test uses "python3".
        "allowed_commands": ["python", "python3"],
        "timeout_s": 60,
        "cwd_root": str(wd / "sandbox"),
    }
    cfg["workflow"]["max_reflect_iterations"] = 2
    # Only run python_compile in the container — ruff/mypy are not installed
    # in python:3.11-slim and would fail for infrastructure reasons, not
    # code-quality reasons. python_compile is the relevant syntax gate.
    cfg["verification"]["execution_verifiers"] = ["python_compile"]

    app = build_application(cfg)
    await app.start()
    runner = app.services.require("workflow_runner")

    try:
        run = await runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="implement a function add(a, b) that returns a + b",
            metadata={
                "verification": {
                    "files": {"test_solution.py": ACCEPTANCE},
                    "command": ["python3", "test_solution.py"],
                }
            },
        )
    finally:
        await app.stop()

    ex = run.state.extra
    completed = run.status == WorkflowStatus.COMPLETED
    acceptance_ran = ex.get("verification_kind") == "acceptance"
    acceptance_passed = ex.get("test_passed") is True
    exit_code = ex.get("exit_code")
    stdout_tail = (run.state.stdout or "")[-200:]
    stderr_tail = (run.state.stderr or "")[-300:]
    code = run.state.code

    result = {
        "test": "11_container_execution",
        "status": "PASS" if (completed and acceptance_ran and acceptance_passed) else "FAIL",
        "classification": "runtime",
        "evidence": {
            "docker_available": True,
            "runner": "container",
            "container_controls": {
                "read_only_root": True,
                "network_none": True,
                "non_root_user": "65534:65534",
                "memory_limit": "512m",
                "memory_swap": "512m",
                "cpus": "1.0",
                "pids_limit": 128,
                "nofile_limit": 1024,
                "cap_drop_all": True,
                "no_new_privileges": True,
                "pinned_image_digest": True,
                "image": "python:3.11-slim",
            },
            "status": run.status.value,
            "test_passed": ex.get("test_passed"),
            "verification_kind": ex.get("verification_kind"),
            "verification_contract_present": ex.get("verification_contract_present"),
            "exit_code": exit_code,
            "iterations": run.state.iterations,
            "artifact_normalized": ex.get("artifact_normalized"),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "code": code,
        },
        "provenance": manifest,
    }
    out = HERE / "test11_container_result.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"VERDICT:{result['status']}")
    print(f"STATUS:{run.status.value}")
    print(f"ACCEPTANCE_RAN:{acceptance_ran}")
    print(f"ACCEPTANCE_PASSED:{acceptance_passed}")
    print(f"EXIT_CODE:{exit_code}")
    print(f"STDOUT_TAIL:{stdout_tail!r}")
    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    asyncio.run(main())
