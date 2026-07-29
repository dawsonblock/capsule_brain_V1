"""Test 9 start phase (process A): start a REAL plan_generate_test_reflect
workflow run on the actual WorkflowRunnerService, then hard-exit the process
while the generate node is still executing (simulating a crash before the
verification/test node).

To make the crash window reliable, the LLM provider used here sleeps for a
long time on the generate call. The workflow CODE is the real built-in
plan_generate_test_reflect workflow — no custom nodes, no changed
implementation. Process B uses the same workflow code with the real local
LLM and resumes into the real acceptance gate.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg
from capsule_brain.events.models import EventEnvelope
from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.builtins import build_plan_generate_test_reflect_workflow

WORKDIR = HERE / "stores" / "t9_recovery"
NONCE = f"ORBIT-RECOVERY-{uuid.uuid4().hex}"
ACCEPTANCE = (
    "from solution import add\n"
    "assert add(2, 3) == 5, 'add(2,3) failed'\n"
    "assert add(-1, 1) == 0, 'add(-1,1) failed'\n"
    "print('ACCEPTANCE_OK')\n"
)


class _SlowGenerateProvider(LLMProvider):
    """Real-ish provider whose generate() sleeps a long time so the process
    can be hard-killed mid-node. The plan call returns quickly so the plan
    node completes and persists; the generate call blocks."""
    name = "slow"

    def __init__(self):
        self.calls = 0

    async def generate(self, request: LLMRequest, model_cfg):
        self.calls += 1
        if self.calls == 1:
            # plan node: return quickly so it persists
            return LLMResult(text="1. Implement add(a,b) returning a+b",
                             model="slow", provider="slow", latency_ms=1)
        # generate node (and any later call): block forever so we can kill
        await asyncio.Event().wait()
        return LLMResult(text="", model="slow", provider="slow", latency_ms=1)

    async def close(self):
        pass


async def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    for f in WORKDIR.glob("*.sqlite"):
        f.unlink(missing_ok=True)

    cfg = base_cfg(WORKDIR)
    cfg["workflow"]["auto_resume"] = False
    # Replace the LLM gateway config to use the slow provider so the generate
    # node blocks and the process can be killed mid-execution.
    cfg["llm_gateway"] = {
        "enable": True,
        "default_model": "slow",
        "timeout_s": 600.0,
        "max_attempts": 1,
        "models": {
            "slow": {
                "provider": "slow",
                "model_name": "slow",
                "capabilities": ["text"],
            }
        },
        "routing": {
            "routes": {
                "coding": {"models": ["slow"]},
                "reflection": {"models": ["slow"]},
            }
        },
    }
    app = build_application(cfg, llm_providers={"slow": _SlowGenerateProvider()})
    await app.start()
    runner = app.services.require("workflow_runner")
    # The bootstrap auto-registers the real plan_generate_test_reflect
    # workflow when workflow.enable=True (using max_reflect_iterations from
    # cfg). It reads services at execution time, so it sees the slow LLM.
    bus = app.services.require("event_bus")

    run_id_box: dict = {}

    async def on_started(event: EventEnvelope):
        run_id_box["id"] = event.payload.get("run_id")
    bus.subscribe("workflow.started", on_started)

    # Start the run in the background; it will block forever at "generate".
    asyncio.create_task(
        runner.start_run(
            workflow_name="plan_generate_test_reflect",
            goal="implement a function add(a, b) that returns a + b",
            metadata={
                "verification": {
                    "files": {"test_solution.py": ACCEPTANCE},
                    "command": ["python3", "test_solution.py"],
                },
                "recovery_nonce": NONCE,
            },
        )
    )

    # Wait for the run id, then for the run to reach the "generate" node
    # (plan must have completed and persisted first).
    for _ in range(400):
        if run_id_box.get("id"):
            break
        await asyncio.sleep(0.05)
    run_id = run_id_box.get("id")
    if not run_id:
        print("ERROR: no run started")
        await app.stop()
        sys.exit(1)

    reached_generate = False
    for _ in range(400):
        run = await runner.repository.get_run(run_id)
        if run is not None and run.current_node == "generate":
            reached_generate = True
            break
        await asyncio.sleep(0.05)

    run = await runner.repository.get_run(run_id)
    if not (run is not None and reached_generate):
        print("ERROR: did not reach generate node")
        await app.stop()
        sys.exit(1)

    # The plan step must be persisted; the generate step is in-flight.
    plan_step_persisted = any(s.node_name == "plan" for s in run.steps)

    print(f"RUN_ID:{run_id}")
    print(f"NONCE:{NONCE}")
    print(f"NODE_AT_CRASH:{run.current_node}")
    print(f"STEPS_BEFORE_CRASH:{len(run.steps)}")
    print(f"PLAN_STEP_PERSISTED:{plan_step_persisted}")
    print(f"WORKFLOW_NAME:{run.workflow_name}")
    sys.stdout.flush()
    # Hard exit simulating a crash: no clean shutdown, generate node left
    # in-flight. The runner's stale-RUNNING->INTERRUPTED conversion on
    # restart will handle this.
    os._exit(2)


if __name__ == "__main__":
    asyncio.run(main())
