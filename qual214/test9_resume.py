"""Test 9 resume phase (process B): start a NEW Capsule Brain process pointing
at the same workflow database. The runner must recognize the interrupted run,
preserve its previous steps and verification contract, and safely continue to
completion."""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowState, WorkflowStatus
from capsule_brain.workflow.runner import Workflow, WorkflowNode

WORKDIR = HERE / "stores" / "t9_recovery"


def completing_recovery_workflow() -> Workflow:
    """Same graph as test9_start, but the 'hold' node is now a no-op so the
    resumed run can complete safely."""

    async def prep(state: WorkflowState) -> WorkflowState:
        return state

    async def hold(state: WorkflowState) -> WorkflowState:
        # No-op on resume: the crash already happened; continue safely.
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
    run_id = os.environ["RECOVERY_RUN_ID"]
    cfg = base_cfg(WORKDIR)
    cfg["workflow"]["auto_resume"] = False  # inspect before resuming
    app = build_application(cfg)
    await app.start()
    runner = app.services.require("workflow_runner")
    runner.register_workflow(completing_recovery_workflow())

    repo = runner.repository
    run = await repo.get_run(run_id)
    if run is None:
        print("VERDICT:FAIL")
        print("ERROR: run not found in new process")
        await app.stop()
        sys.exit(1)

    interrupted_recognized = run.status == WorkflowStatus.INTERRUPTED
    same_run_id = run.id == run_id
    steps_before = len(run.steps)
    contract = (run.metadata or {}).get("verification")
    contract_preserved = bool(contract and contract.get("command")
                              and contract.get("files"))
    nonce_preserved = bool((run.metadata or {}).get("recovery_nonce"))
    prep_step_preserved = any(s.node_name == "prep" for s in run.steps)

    print(f"SAME_RUN_ID:{same_run_id}")
    print(f"STATUS_BEFORE_RESUME:{run.status.value}")
    print(f"INTERRUPTED_RECOGNIZED:{interrupted_recognized}")
    print(f"STEPS_PRESERVED:{steps_before}")
    print(f"PREP_STEP_PRESERVED:{prep_step_preserved}")
    print(f"CONTRACT_PRESERVED:{contract_preserved}")
    print(f"NONCE_PRESERVED:{nonce_preserved}")

    # Safe continuation: explicitly resume the interrupted run.
    resumed = await runner.resume_run(run_id)
    final = await repo.get_run(run_id)
    completed = final.status == WorkflowStatus.COMPLETED
    print(f"STATUS_AFTER_RESUME:{final.status.value}")
    print(f"STEPS_AFTER_RESUME:{len(final.steps)}")
    print(f"COMPLETED:{completed}")

    await app.stop()
    ok = (same_run_id and interrupted_recognized and prep_step_preserved
          and contract_preserved and nonce_preserved and completed)
    print("VERDICT:PASS" if ok else "VERDICT:FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
