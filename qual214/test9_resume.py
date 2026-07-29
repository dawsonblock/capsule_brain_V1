"""Test 9 resume phase (process B): start a NEW Capsule Brain process pointing
at the same workflow database, using the SAME real plan_generate_test_reflect
workflow code and the REAL local LLM. The runner must recognize the
interrupted run, preserve its previous steps and verification contract, and
safely resume — re-running the generate node with the real LLM and then
running the REAL acceptance gate. The run must reach COMPLETED only after
acceptance passes.
"""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg, QWEN_CODER
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowStatus

WORKDIR = HERE / "stores" / "t9_recovery"


async def main():
    run_id = os.environ["RECOVERY_RUN_ID"]
    cfg = base_cfg(WORKDIR, model=QWEN_CODER)
    cfg["workflow"]["auto_resume"] = False  # inspect before resuming
    cfg["workflow"]["max_reflect_iterations"] = 2
    app = build_application(cfg)
    await app.start()
    runner = app.services.require("workflow_runner")
    # The bootstrap auto-registers the SAME real plan_generate_test_reflect
    # workflow used by process A (same code, max_reflect_iterations=2 from
    # cfg). It reads services at execution time, so it sees the real LLM.

    repo = runner.repository
    run = await repo.get_run(run_id)
    if run is None:
        print("VERDICT:FAIL")
        print("ERROR: run not found in new process")
        await app.stop()
        sys.exit(1)

    interrupted_recognized = run.status == WorkflowStatus.INTERRUPTED
    same_run_id = run.id == run_id
    same_workflow = run.workflow_name == "plan_generate_test_reflect"
    steps_before = len(run.steps)
    plan_step_preserved = any(s.node_name == "plan" for s in run.steps)
    contract = (run.metadata or {}).get("verification")
    contract_preserved = bool(contract and contract.get("command")
                              and contract.get("files"))
    nonce_preserved = bool((run.metadata or {}).get("recovery_nonce"))
    plan_text_preserved = bool(run.state.plan)

    print(f"SAME_RUN_ID:{same_run_id}")
    print(f"SAME_WORKFLOW:{same_workflow}")
    print(f"STATUS_BEFORE_RESUME:{run.status.value}")
    print(f"INTERRUPTED_RECOGNIZED:{interrupted_recognized}")
    print(f"STEPS_PRESERVED:{steps_before}")
    print(f"PLAN_STEP_PRESERVED:{plan_step_preserved}")
    print(f"PLAN_TEXT_PRESERVED:{plan_text_preserved}")
    print(f"CONTRACT_PRESERVED:{contract_preserved}")
    print(f"NONCE_PRESERVED:{nonce_preserved}")

    # Safe continuation: explicitly resume the interrupted run. The generate
    # node re-runs with the REAL local LLM, then the test node runs the REAL
    # acceptance gate.
    resumed = await runner.resume_run(run_id)
    final = await repo.get_run(run_id)
    ex = final.state.extra
    completed = final.status == WorkflowStatus.COMPLETED
    acceptance_ran = ex.get("verification_kind") == "acceptance"
    acceptance_passed = ex.get("test_passed") is True
    acceptance_contract_present = ex.get("verification_contract_present") is True
    exit_code = ex.get("exit_code")

    print(f"STATUS_AFTER_RESUME:{final.status.value}")
    print(f"STEPS_AFTER_RESUME:{len(final.steps)}")
    print(f"ACCEPTANCE_RAN:{acceptance_ran}")
    print(f"ACCEPTANCE_PASSED:{acceptance_passed}")
    print(f"ACCEPTANCE_CONTRACT_PRESENT:{acceptance_contract_present}")
    print(f"EXIT_CODE:{exit_code}")
    print(f"ITERATIONS:{final.state.iterations}")
    print(f"FINAL_CODE:{(final.state.code or '')[:80]!r}")
    print(f"STDOUT_TAIL:{(final.state.stdout or '')[-80:]!r}")

    await app.stop()
    # PASS requires: same run/workflow, interrupted recognized, plan step +
    # contract preserved, AND the real acceptance gate ran and passed on
    # resume (COMPLETED only after acceptance).
    ok = (same_run_id and same_workflow and interrupted_recognized
          and plan_step_preserved and contract_preserved and nonce_preserved
          and completed and acceptance_ran and acceptance_passed
          and acceptance_contract_present)
    print("VERDICT:PASS" if ok else "VERDICT:FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
