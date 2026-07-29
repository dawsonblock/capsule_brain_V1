# Capsule Brain v2.13.1 — Verification-Coverage Completion

v2.13.1 is a regression-coverage and release-qualification pass over the v2.13.0 verification-boundary repair. No production logic in `src/` was changed. The source tree remains byte-identical to v2.13.0; this release adds the missing mandatory tests, records executed release evidence, and bumps package metadata to 2.13.1.

## Scope completed

Four named regression additions close the verification gaps identified after v2.13.0:

1. `test_acceptance_verification_rejects_path_traversal`
   - Parametrized over `../evil.py`, an absolute path, and the reserved `solution.py` collision.
   - Confirms the run fails closed, `test_passed` remains false, no escaped file is created, and the per-run acceptance directory is cleaned.

2. `test_failing_acceptance_script_blocks_completion_despite_clean_solution`
   - Uses a valid, cleanly exiting `solution.py` but an intentionally failing acceptance script.
   - Confirms the acceptance command's non-zero exit code blocks completion.

3. `test_acceptance_verification_fails_closed_on_execution_infrastructure_error`
   - Exercises two infrastructure-failure stages in one regression: an acceptance executor exception (`daemon unreachable`) and a shared verification-service exception (`verifier backend unavailable`).
   - Both paths terminate without promotion and preserve diagnostic failure text.

4. `test_build_application_rejects_smoke_workflow_without_acknowledgement`
   - Confirms construction-boundary config validation rejects `workflow.enable=true` plus `verification_mode=smoke` unless `allow_smoke_success=true` is explicitly set.

Because the traversal test has three parametrized cases, the four named additions contribute six collected pytest cases. The full suite therefore increases from 176 to 182 tests.

## Production-source integrity

No `src/` change was required. A SHA-256 tree comparison excluding generated caches/metadata produced the same digest for v2.13.0 and v2.13.1 source code:

```text
bcfc46a22aaa84f87de6d051bb6987b34b346fa4168a7fdd92e4abd09f4d2c6d
```

## Executed validation results

The following commands were executed against the final 2.13.1 tree:

```bash
python -m compileall -q src tests
pytest -q
PYTHONPATH=src python demo.py
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Observed results:

- `compileall`: PASS, exit code 0.
- Full pytest suite: PASS — **182 passed in 16.78s**.
- Demo: PASS — application started, conversation/feedback/reflection/verification/goal flow executed, and all services stopped cleanly.
- Wheel build: PASS — `capsule_brain-2.13.1-py3-none-any.whl` created successfully.
- Wheel SHA-256 at build time: `4ab889a624aec75a8d4493a488e2dce809f9c0a15f740d2ae29b9f46b4c88a55`.

## Isolated wheel import check

A clean venv was created and the freshly built wheel installed with `--no-deps`. Direct dependency installation from the configured package index and from `https://pypi.org/simple` could not be completed because this execution environment has no outbound package-resolution/DNS access. This is an environment limitation, not a package test pass and is recorded as such.

To still qualify the wheel itself without using the repository source tree, the exact dependency closure already installed in the qualified host environment was copied into a clean venv. The wheel remained the only Capsule Brain package installed there. The following import surfaces then passed:

```python
from capsule_brain.runtime.bootstrap import build_application
from capsule_brain.workflow.models import WorkflowStatus
from capsule_brain.llm.tools import ToolRegistry
```

Observed package versions in that isolated check:

```text
capsule-brain 2.13.1
httpx 0.28.1
PyYAML 6.0.3
jsonschema 4.26.0
```

`pip check` in that clean, locally provisioned environment reported:

```text
No broken requirements found.
```

A future network-enabled CI job should repeat normal dependency resolution from a package index. No claim is made here that outbound-network installation was executed successfully.

## Version

The authoritative package declaration in `pyproject.toml` is now:

```toml
version = "2.13.1"
```

Historical v2.13.0 milestone, validation, and repair-prompt documents are retained unchanged as release history.

## Deliberately not claimed

v2.13.0 is not a self-learning release. Experience collection is improved, but there is still no learned executive policy, weight update path, causal world model, or online policy promotion mechanism.
