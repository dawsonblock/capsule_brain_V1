"""Test 9 start phase (process A): start a REAL workflow run on the actual
WorkflowRunnerService, let it persist the first node, then hard-exit the
process while the second node is still executing (simulating a crash)."""
import asyncio
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg
from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowState
from capsule_brain.workflow.runner import Workflow, WorkflowNode, WorkflowRunnerService

WORKDIR = HERE / "stores" / "t9_recovery"
NONCE = f"ORBIT-RECOVERY-{uuid.uuid4().hex}"
VERIFICATION_CONTRACT = {
    "files": {"test_solution.py": "from solution import add\nassert add(2,3)==5\n"},
    "command": ["python3", "test_solution.py"],
}


def blocking_recovery_workflow() -> Workflow:
    """A -> hold(blocks forever) -> finish. The hold node blocks on an event
    that is never set, so the process must be killed to leave the run
    mid-execution."""

    async def prep(state: WorkflowState) -> WorkflowState:
        state.plan = "prep done; contract stored"
        state.extra["recovery_nonce"] = NONCE
        return state

    async def hold(state: WorkflowState) -> WorkflowState:
        # Block forever. The process will be hard-killed here.
        await asyncio.Event().wait()
        return state

    async def finish(state: WorkflowState) -> WorkflowState:
        state.extra["finished"] = True
        return state

    return Workflow(
        name="recovery_demo",
        nodes=[
            WorkflowNode("prep", prep, next_node="hold"),
            WorkflowNode("hold", hold, next_node="finish"),
            WorkflowNode("finish", finish, next_node=None),
        ],
        entry="prep",
    )


async def main():
    # Clean prior DB for determinism.
    WORKDIR.mkdir(parents=True, exist_ok=True)
    for f in WORKDIR.glob("*.sqlite"):
        f.unlink(missing_ok=True)

    cfg = base_cfg(WORKDIR)
    cfg["workflow"]["auto_resume"] = False
    app = build_application(cfg)
    await app.start()
    runner = app.services.require("workflow_runner")
    bus = app.services.require("event_bus")
    runner.register_workflow(blocking_recovery_workflow())

    run_id_box: dict = {}

    async def on_started(event: EventEnvelope):
        run_id_box["id"] = event.payload.get("run_id")
    bus.subscribe("workflow.started", on_started)

    # Start the run in the background; it will block forever at "hold".
    asyncio.create_task(
        runner.start_run(
            workflow_name="recovery_demo",
            goal="recovery demonstration",
            metadata={"verification": VERIFICATION_CONTRACT, "recovery_nonce": NONCE},
        )
    )

    # Wait for the run id, then for the run to reach the "hold" node.
    for _ in range(200):
        if run_id_box.get("id"):
            break
        await asyncio.sleep(0.05)
    run_id = run_id_box.get("id")
    if not run_id:
        print("ERROR: no run started")
        await app.stop()
        sys.exit(1)

    for _ in range(200):
        run = await runner.repository.get_run(run_id)
        if run is not None and run.current_node == "hold":
            break
        await asyncio.sleep(0.05)

    run = await runner.repository.get_run(run_id)
    assert run is not None and run.current_node == "hold", "did not reach hold"
    assert len(run.steps) >= 1, "prep step not persisted"

    print(f"RUN_ID:{run_id}")
    print(f"NONCE:{NONCE}")
    print(f"NODE_AT_CRASH:{run.current_node}")
    print(f"STEPS_BEFORE_CRASH:{len(run.steps)}")
    sys.stdout.flush()
    # Hard exit simulating a crash: no clean shutdown, step left "running".
    os._exit(2)


if __name__ == "__main__":
    asyncio.run(main())
