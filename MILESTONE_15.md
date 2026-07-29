# Capsule Brain v2.14.0 — Connected Build

v2.14.0 is the connected-build release that makes the source tree, the built
wheel, the tests, and the live local-LLM qualification evidence all refer to
the same immutable build. It supersedes the v2.13.1 source tree and brings the
two production-source corrections that were previously only present in the
external v2.14.0 wheel into `src/`.

## Scope

Three production-source corrections (the only files that differ from v2.13.1):

1. `runtime/bootstrap.py` — construction-boundary rejection of the
   `<section>.enabled` configuration typo. The runtime uses `enable`
   consistently for service activation; silently accepting `enabled` previously
   let a qualification run appear to boot correctly while major services were
   actually absent. The validator now rejects the alias with a corrective hint
   for every service section: `llm_gateway`, `conversation`, `feedback`,
   `reflection`, `execution`, `verification`, `workflow`, `redis_bridge`,
   `memory_consolidator`.

2. `workflow/builtins.py` — `normalize_python_artifact` conservatively strips a
   single outer Markdown fence from a generated Python artifact before static
   verification. Small local chat models frequently wrap valid Python in a
   code fence; that is a transport/presentation defect, not a semantic
   programming defect, and it should not consume the reflection budget. Only an
   *outer* fence is removed; nested fences or prose trailers are left untouched
   so verification still fails closed.

3. `workflow/builtins.py` — the reflect node's repair prompt is now
   verifier-conditioned. It feeds the verification kind and the verifier /
   acceptance failure text into the repair request and instructs the model to
   fix the specific failure, preserve required signatures, return raw Python
   source only, and not weaken or bypass the acceptance test. v2.13.1 used a
   much weaker generic "produce corrected code" prompt.

No other production source was changed. The remainder of the source tree is
byte-identical to v2.13.1.

## Provenance integrity

The v2.13.1→v2.14.0 source correction was verified by copying the two
differing files from the authoritative installed v2.14.0 wheel into `src/` and
confirming:

- `diff -rq src/capsule_brain <installed v2.14.0 package>` produces no
  differences (byte-identical package contents).
- A wheel built from the corrected `src/` installs as `capsule-brain 2.14.0`
  with `normalize_python_artifact` present.
- The source-built wheel's installed package contents are byte-identical to the
  original v2.14.0 wheel's installed package contents.

Wheel SHA-256 values differ because ZIP archives embed build metadata/timestamps;
the authoritative equivalence check is the installed-package byte comparison,
which is identical.

## Executed validation results

The following commands were executed against the corrected 2.14.0 tree:

```bash
python -m compileall -q src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python demo.py
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Observed results:

- `compileall`: PASS, exit code 0.
- Full pytest suite: PASS — **199 passed in 6.04s** (182 pre-existing + 17 new
  `test_v214_connected.py` cases pinning the three corrections).
- Demo: PASS — application started, conversation/feedback/reflection/
  verification/goal flow executed, all services stopped cleanly.
- Wheel build: PASS — `capsule_brain-2.14.0-py3-none-any.whl` created.

## Live local-LLM qualification

A separate live qualification harness (`qual214/`) exercises the real installed
runtime against a small local LLM (Ollama 0.12.0, `qwen2.5-coder:3b` primary,
`llama3.1:8b` secondary for native tool-calling). The harness now enforces
provenance: it asserts `importlib.metadata.version("capsule-brain") == "2.14.0"`
and records the wheel SHA-256, source-tree SHA-256, Python version, platform,
Ollama version, and model digests alongside the results. See
`qual214/LOCAL_LLM_QUALIFICATION.md` and `qual214/qualification_results.json`.

## Version

The authoritative package declaration in `pyproject.toml` is now:

```toml
version = "2.14.0"
```

## Deliberately not claimed

v2.14.0 is not a self-learning release. Experience collection, feedback,
verification outcomes, tool telemetry, and reflection results are persisted,
but there is no policy learner, reward model, router/adapter/LoRA/steering-vector
update, preference learner, world-model update, or online policy promotion
mechanism. Experience collection is a data substrate, not learning.
