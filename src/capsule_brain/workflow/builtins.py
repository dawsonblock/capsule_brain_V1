from __future__ import annotations

import logging
from typing import Any

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest

from .models import WorkflowState
from .runner import Workflow, WorkflowNode

log = logging.getLogger(__name__)

# Cap reflect->generate loops so a stuck workflow cannot loop forever.
# The runner's max_steps is the hard cap; this is the soft cap specific to
# the plan->generate->test->reflect cycle.
DEFAULT_MAX_REFLECT_ITERATIONS = 3


def normalize_python_artifact(text: str) -> tuple[str, bool]:
    """Normalize common presentation wrappers around a Python artifact.

    Small/local chat models frequently obey the code-generation request but
    still wrap the entire artifact in a single Markdown code fence.  That is a
    transport/presentation defect, not a semantic programming defect, and it
    should not consume the reflection budget.  Only an *outer* fence is removed;
    arbitrary prose or embedded fences are left untouched so verification still
    fails closed.

    Returns ``(normalized_text, changed)``.
    """
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw, False
    lines = raw.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return raw, False
    opener = lines[0].strip().lower()
    if opener not in {"```", "```python", "```py"}:
        return raw, False
    inner = "\n".join(lines[1:-1]).strip()
    # Refuse to normalize nested fences or prose-like trailing wrappers.
    if "```" in inner:
        return raw, False
    return inner, inner != raw


def _safe_get(services: Any, name: str) -> Any:
    """Fetch a service from a registry/dict, returning None if absent."""
    if services is None:
        return None
    if hasattr(services, "get"):
        return services.get(name)
    if isinstance(services, dict):
        return services.get(name)
    return None


def build_plan_generate_test_reflect_workflow(
    *,
    services: Any,
    max_iterations: int = DEFAULT_MAX_REFLECT_ITERATIONS,
    require_approval: bool = False,
    verification_mode: str = "acceptance",
    allow_smoke_success: bool = False,
) -> Workflow:
    """Build the default Plan -> Generate -> Test -> Reflect workflow.

    Nodes:
    - ``plan``: ask the LLM to decompose the goal into a concrete plan.
    - ``generate``: ask the LLM to produce code from the plan.
    - ``test``: verify generated code. Safe ``acceptance`` mode requires a
      caller-supplied verification command/assets; legacy ``smoke`` mode is
      available only behind an explicit opt-in.
    - ``approve``: optional human-in-the-loop checkpoint. Pauses before
      completing so an operator can review the generated code. Enabled when
      ``require_approval=True``.
    - ``reflect``: if the test failed and iterations remain, loop back to
      ``generate`` with the failure context. Otherwise, complete.

    Safety semantics are fail-closed:
    - No LLMGateway: generation is marked failed; placeholder text can never
      be promoted as successfully generated code.
    - No ExecutionService: the artifact is unverified and the test node fails.
    - No acceptance contract in the default mode: the artifact is unverified
      and the test node fails rather than equating exit code 0 with correctness.
    - Exhausting the reflection budget terminates the run as FAILED rather
      than converting repeated test failures into completion.

    Args:
        services: a ServiceRegistry or dict providing ``llm_gateway`` and
            ``execution`` services. Either may be absent.
        max_iterations: max reflect->generate loops before giving up.
        require_approval: if True, insert an approval checkpoint before
            completion.
    """
    verification_mode = str(verification_mode).strip().lower()
    if verification_mode not in {"acceptance", "smoke"}:
        raise ValueError(
            "verification_mode must be 'acceptance' or 'smoke'"
        )
    if verification_mode == "smoke" and not allow_smoke_success:
        raise ValueError(
            "smoke verification can only promote a workflow when "
            "allow_smoke_success=True is explicitly set"
        )

    llm = _safe_get(services, "llm_gateway")
    execution = _safe_get(services, "execution")
    verification = _safe_get(services, "verification")

    async def plan_node(state: WorkflowState) -> WorkflowState:
        if llm is None or not llm.is_running:
            state.plan = f"[offline] Plan steps for: {state.goal}"
            return state
        try:
            result = await llm.generate(
                LLMRequest(
                    prompt=(
                        f"Decompose this goal into a concise, actionable plan "
                        f"for code generation. Output the plan only.\n\n"
                        f"GOAL: {state.goal}"
                    ),
                    temperature=0.2,
                    system=(
                        "You are a planning module for an autonomous coding "
                        "agent. Produce a short, concrete plan."
                    ),
                ),
                route="coding",
            )
            state.plan = result.text.strip()
        except Exception as exc:
            log.warning("plan node LLM call failed: %s", exc)
            state.plan = f"[plan failed: {exc}] {state.goal}"
        return state

    async def generate_node(state: WorkflowState) -> WorkflowState:
        if llm is None or not llm.is_running:
            state.code = ""
            state.extra["generation_failed"] = True
            state.errors.append("generation unavailable: llm_gateway is not running")
            return state
        try:
            prompt = (
                f"PLAN:\n{state.plan}\n\n"
                f"GOAL:\n{state.goal}\n\n"
            )
            if state.errors:
                prompt += (
                    f"PREVIOUS ERRORS (fix these):\n"
                    f"{chr(10).join(state.errors[-3:])}\n\n"
                )
            prompt += "Output ONLY the Python code, no markdown fences."
            result = await llm.generate(
                LLMRequest(
                    prompt=prompt,
                    temperature=0.0,
                    system=(
                        "You are a code generation module. Output only "
                        "valid, runnable Python code."
                    ),
                ),
                route="coding",
            )
            state.code, normalized = normalize_python_artifact(result.text)
            state.extra["artifact_normalized"] = bool(normalized)
            state.extra["generation_failed"] = not bool(state.code)
            if not state.code:
                state.errors.append("generation returned empty code")
        except Exception as exc:
            log.warning("generate node LLM call failed: %s", exc)
            state.code = ""
            state.extra["generation_failed"] = True
            state.errors.append(f"generation failed: {exc}")
        return state

    async def test_node(state: WorkflowState) -> WorkflowState:
        """Verify the generated artifact.

        ``acceptance`` mode is the safe default. It requires an externally
        supplied verification contract in workflow run metadata::

            {
                "verification": {
                    "files": {"test_solution.py": "..."},
                    "command": ["pytest", "-q", "test_solution.py"]
                }
            }

        The generated artifact is written as ``solution.py`` beside those
        files in a run-specific sandbox directory. A zero exit code from the
        generated program itself is never treated as proof of goal success.

        ``smoke`` mode preserves the legacy run-the-artifact behavior, but it
        is available only behind the explicit ``allow_smoke_success`` opt-in.
        """
        if state.extra.get("generation_failed") or not state.code.strip():
            state.stdout = ""
            state.stderr = "[generation failed: no executable artifact]"
            state.extra["exit_code"] = None
            state.extra["test_passed"] = False
            state.extra["verification_kind"] = verification_mode
            return state
        if execution is None or not execution.is_running:
            state.stdout = ""
            state.stderr = "[execution unavailable: artifact not verified]"
            state.extra["exit_code"] = None
            state.extra["test_passed"] = False
            state.extra["verification_kind"] = verification_mode
            return state

        # Run the shared verification service first when available. This
        # provides syntax/compile checks and records an auditable verifier run.
        if verification is not None and verification.is_running:
            try:
                from capsule_brain.verification.models import VerificationStatus

                verification_result = await verification.verify(
                    subject=state.code,
                    source="workflow",
                    metadata={
                        "content_type": "python",
                        "goal": state.goal,
                        "workflow_run_id": state.run_id,
                    },
                    publish_events=False,
                )
                state.extra["static_verification_id"] = verification_result.id
                state.extra["static_verification_status"] = (
                    verification_result.status.value
                )
                if verification_result.status == VerificationStatus.FAIL:
                    failures = [
                        check.summary
                        for check in verification_result.checks
                        if check.status == VerificationStatus.FAIL
                    ]
                    state.stdout = ""
                    state.stderr = "[static verification failed] " + "; ".join(failures)
                    state.extra["exit_code"] = None
                    state.extra["test_passed"] = False
                    state.extra["verification_kind"] = "static"
                    return state
            except Exception as exc:
                # Verification infrastructure failure is not a pass. Fail
                # closed and let the reflection loop decide whether to retry.
                log.warning("workflow static verification failed: %s", exc)
                state.stdout = ""
                state.stderr = f"[verification service failed: {exc}]"
                state.extra["exit_code"] = None
                state.extra["test_passed"] = False
                state.extra["verification_kind"] = "static"
                return state

        if verification_mode == "smoke":
            return await _run_smoke_verification(state)
        return await _run_acceptance_verification(state)

    async def _run_smoke_verification(state: WorkflowState) -> WorkflowState:
        """Legacy execution smoke test, gated behind explicit unsafe opt-in."""
        tmp_name: str | None = None
        try:
            from capsule_brain.execution.models import ExecutionRequest
            import os
            import tempfile

            cwd_root = execution.policy.cwd_root
            os.makedirs(cwd_root, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                dir=cwd_root,
                prefix="wf_",
                encoding="utf-8",
            )
            tmp.write(state.code)
            tmp.close()
            tmp_name = tmp.name

            interpreter = next(
                (c for c in execution.policy.allowed_commands if "python" in c.lower()),
                "python",
            )
            result = await execution.execute(
                ExecutionRequest(
                    command=[interpreter, os.path.basename(tmp_name)],
                    cwd=cwd_root,
                    source="workflow:smoke",
                    metadata={"run_id": state.run_id, "verification": "smoke"},
                )
            )
            state.stdout = result.stdout
            state.stderr = result.stderr
            state.extra["exit_code"] = result.exit_code
            state.extra["test_passed"] = bool(result.passed)
            state.extra["verification_kind"] = "smoke"
            return state
        except Exception as exc:
            log.warning("smoke verification failed: %s", exc)
            state.stderr = f"[smoke verification failed: {exc}]"
            state.extra["exit_code"] = None
            state.extra["test_passed"] = False
            state.extra["verification_kind"] = "smoke"
            return state
        finally:
            if tmp_name:
                try:
                    import os
                    os.unlink(tmp_name)
                except OSError:
                    pass

    async def _run_acceptance_verification(state: WorkflowState) -> WorkflowState:
        """Run caller-supplied acceptance assets against ``solution.py``."""
        from capsule_brain.execution.models import ExecutionRequest
        from pathlib import Path
        import shutil

        request_metadata = state.extra.get("request_metadata") or {}
        contract = request_metadata.get("verification") or {}
        files = contract.get("files") or {}
        command = contract.get("command") or []

        if not isinstance(files, dict) or not isinstance(command, list) or not command:
            state.stdout = ""
            state.stderr = (
                "[acceptance verification missing] supply workflow metadata "
                "verification.files and verification.command; execution success "
                "alone is not proof that the goal was solved"
            )
            state.extra["exit_code"] = None
            state.extra["test_passed"] = False
            state.extra["verification_kind"] = "acceptance"
            state.extra["verification_contract_present"] = False
            return state

        root = Path(execution.policy.cwd_root).resolve()
        run_dir = (root / f"workflow_{state.run_id}").resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            state.stderr = "[acceptance verification refused: run directory escaped sandbox]"
            state.extra["test_passed"] = False
            return state

        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "solution.py").write_text(state.code, encoding="utf-8")
            for raw_name, content in files.items():
                name = str(raw_name)
                destination = (run_dir / name).resolve()
                try:
                    destination.relative_to(run_dir)
                except ValueError:
                    raise ValueError(f"verification file escapes run directory: {name!r}")
                if destination == run_dir or name in {"", ".", "solution.py"}:
                    raise ValueError(f"invalid verification filename: {name!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(str(content), encoding="utf-8")

            result = await execution.execute(
                ExecutionRequest(
                    command=[str(item) for item in command],
                    cwd=str(run_dir),
                    source="workflow:acceptance",
                    metadata={
                        "run_id": state.run_id,
                        "verification": "acceptance",
                        "goal": state.goal,
                    },
                )
            )
            state.stdout = result.stdout
            state.stderr = result.stderr
            state.extra["exit_code"] = result.exit_code
            state.extra["test_passed"] = bool(result.passed)
            state.extra["verification_kind"] = "acceptance"
            state.extra["verification_contract_present"] = True
            state.extra["verification_command"] = [str(item) for item in command]
            return state
        except Exception as exc:
            log.warning("acceptance verification failed: %s", exc)
            state.stdout = ""
            state.stderr = f"[acceptance verification failed: {exc}]"
            state.extra["exit_code"] = None
            state.extra["test_passed"] = False
            state.extra["verification_kind"] = "acceptance"
            state.extra["verification_contract_present"] = True
            return state
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_passed(state: WorkflowState) -> bool:
        return bool(state.extra.get("test_passed", False))

    def after_test(state: WorkflowState) -> str | None:
        """Conditional edge after the test node.

        - If the test passed, go to approve (if configured) or complete.
        - If the test failed and iterations remain, go to reflect.
        - If the test failed and the budget is exhausted, enter an explicit
          failure node.
        """
        if test_passed(state):
            return "approve" if require_approval else "complete"
        if state.iterations < max_iterations:
            return "reflect"
        return "failed"

    async def reflect_node(state: WorkflowState) -> WorkflowState:
        state.iterations += 1
        # Record the failure context for the next generate iteration.
        failure = (
            f"Iteration {state.iterations}: test failed. "
            f"stderr: {state.stderr[:500]}"
        )
        state.errors.append(failure)
        if llm is not None and llm.is_running:
            try:
                result = await llm.generate(
                    LLMRequest(
                        prompt=(
                            f"GOAL:\n{state.goal}\n\n"
                            f"PLAN:\n{state.plan}\n\n"
                            f"CURRENT PYTHON ARTIFACT:\n{state.code}\n\n"
                            f"VERIFICATION KIND: "
                            f"{state.extra.get('verification_kind', 'unknown')}\n"
                            f"VERIFIER / ACCEPTANCE FAILURE:\n"
                            f"{state.stderr[:1400]}\n\n"
                            "REPAIR REQUIREMENTS:\n"
                            "- Fix the specific verifier or acceptance failure above.\n"
                            "- Preserve required public function/class signatures.\n"
                            "- Return raw Python source only.\n"
                            "- Do not include Markdown fences or explanations.\n"
                            "- Do not weaken or bypass the acceptance test.\n"
                        ),
                        temperature=0.0,
                        system=(
                            "You are a verifier-conditioned code repair module. "
                            "The external verifier is authoritative. Produce only "
                            "the corrected Python artifact."
                        ),
                    ),
                    route="reflection",
                )
                state.code, normalized = normalize_python_artifact(result.text)
                state.extra["artifact_normalized"] = bool(normalized)
                state.extra["generation_failed"] = not bool(state.code)
            except Exception as exc:
                log.warning("reflect node LLM call failed: %s", exc)
                state.errors.append(f"reflect failed: {exc}")
        return state

    def after_reflect(state: WorkflowState) -> str:
        # Always loop back to test after reflect. The after_test edge will
        # decide whether to reflect again or complete.
        return "test"

    async def approve_node(state: WorkflowState) -> WorkflowState:
        # The checkpoint guard on this node handles the pause. The action
        # itself is a no-op; the runner parks before transitioning to
        # ``complete``.
        return state

    def approve_checkpoint(state: WorkflowState) -> bool:
        # Always pause at the approve node so the operator can review.
        return True

    async def complete_node(state: WorkflowState) -> WorkflowState:
        # Terminal success node. It is reachable only after a verified pass.
        return state

    async def failed_node(state: WorkflowState) -> WorkflowState:
        raise RuntimeError(
            f"verification failed after {state.iterations} reflection iterations"
        )

    nodes = [
        WorkflowNode(
            "plan",
            plan_node,
            next_node="generate",
            description="Decompose the goal into a plan",
        ),
        WorkflowNode(
            "generate",
            generate_node,
            next_node="test",
            description="Generate code from the plan",
        ),
        WorkflowNode(
            "test",
            test_node,
            next_node=after_test,
            description="Run the generated code and capture results",
        ),
        WorkflowNode(
            "reflect",
            reflect_node,
            next_node=after_reflect,
            description="Reflect on test failure and produce a fix",
        ),
        WorkflowNode(
            "approve",
            approve_node,
            next_node="complete",
            checkpoint=approve_checkpoint if require_approval else None,
            description="Human-in-the-loop approval checkpoint",
        ),
        WorkflowNode(
            "failed",
            failed_node,
            next_node=None,
            description="Terminal failure after verifier budget exhaustion",
        ),
        WorkflowNode(
            "complete",
            complete_node,
            next_node=None,
            description="Terminal node",
        ),
    ]

    return Workflow(
        name="plan_generate_test_reflect",
        nodes=nodes,
        entry="plan",
        description=(
            "Plan -> Generate -> Test -> (Pass? -> Approve? -> Complete | "
            "Fail? -> Reflect -> loop to Test)"
        ),
    )
