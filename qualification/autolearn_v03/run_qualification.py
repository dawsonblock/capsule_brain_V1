"""Single-command qualification pipeline for AutoLearn v2.16.0.

Priority 11: Runs the entire qualification pipeline in one command:

1. run_counterfactuals.py  — generate counterfactual outcomes
2. build_dataset.py        — build the supervised dataset
3. train_policy.py         — train the candidate policy
4. train_sham.py           — train the sham-learning control
5. evaluate_policy.py      — evaluate candidate vs baseline
6. evaluate_sham.py        — evaluate sham vs baseline
7. run_shadow_eval.py      — shadow evaluation
8. promote_policy.py       — promotion gate (15 gates)
9. generate_artifacts.py   — generate all artifacts
10. report.py              — consolidated report

Usage:
    python run_qualification.py --runtime real --epsilon 0.5
    python run_qualification.py --runtime simulated  # for testing only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_step(name: str, cmd: list[str], *, env: dict | None = None) -> bool:
    """Run a pipeline step and return True if it succeeded."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{name}] {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full AutoLearn v2.16.0 qualification pipeline."
    )
    parser.add_argument(
        "--runtime",
        choices=["real", "simulated"],
        default="real",
        help="Evaluation runtime: 'real' (default, official) or 'simulated'.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Minimum effect threshold for promotion gate. Default 0.0.",
    )
    parser.add_argument(
        "--skip-shadow",
        action="store_true",
        help="Skip shadow evaluation (for faster iteration).",
    )
    parser.add_argument(
        "--skip-sham",
        action="store_true",
        help="Skip sham-learning control (for faster iteration).",
    )
    args = parser.parse_args()

    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    # Step 1: Counterfactuals
    steps.append(("counterfactuals", [py, "run_counterfactuals.py", "--runtime", args.runtime]))

    # Step 2: Build dataset
    steps.append(("build_dataset", [py, "build_dataset.py"]))

    # Step 3: Train candidate policy
    steps.append(("train_policy", [py, "train_policy.py"]))

    # Step 4: Train sham policy (Priority 4)
    if not args.skip_sham:
        steps.append(("train_sham", [py, "train_sham.py"]))

    # Step 5: Evaluate candidate
    steps.append(("evaluate_policy", [py, "evaluate_policy.py", "--runtime", args.runtime]))

    # Step 6: Evaluate sham (Priority 4)
    if not args.skip_sham:
        steps.append(("evaluate_sham", [py, "evaluate_sham.py", "--runtime", args.runtime]))

    # Step 7: Shadow evaluation
    if not args.skip_shadow:
        steps.append(("shadow_eval", [py, "run_shadow_eval.py"]))

    # Step 8: Promotion gate
    promote_cmd = [py, "promote_policy.py", "--epsilon", str(args.epsilon)]
    steps.append(("promote_policy", promote_cmd))

    # Step 9: Generate artifacts
    steps.append(("generate_artifacts", [py, "generate_artifacts.py"]))

    # Step 10: Report
    steps.append(("report", [py, "report.py"]))

    # Run all steps.
    print(f"AutoLearn v2.16.0 Qualification Pipeline")
    print(f"  runtime: {args.runtime}")
    print(f"  epsilon: {args.epsilon}")
    print(f"  skip_shadow: {args.skip_shadow}")
    print(f"  skip_sham: {args.skip_sham}")
    print(f"  steps: {len(steps)}")

    failed: list[str] = []
    for name, cmd in steps:
        ok = _run_step(name, cmd)
        if not ok:
            failed.append(name)
            print(f"\nERROR: step '{name}' failed. Stopping pipeline.")
            break

    if failed:
        print(f"\n{'='*60}")
        print(f"PIPELINE FAILED: {', '.join(failed)}")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE: all {len(steps)} steps succeeded")
        print(f"{'='*60}")
        # Print promotion result.
        prom_path = Path(__file__).resolve().parent / "promotion_result.json"
        if prom_path.exists():
            import json
            prom = json.loads(prom_path.read_text())
            print(f"\nPromotion: {'PROMOTED' if prom.get('promoted') else 'REJECTED'}")
            print(f"Reason: {prom.get('gate_result', {}).get('reason', 'unknown')}")
            print(f"Runtime: {prom.get('runtime_type', 'unknown')}")
            print(f"Epsilon: {prom.get('epsilon', 0.0)}")


if __name__ == "__main__":
    main()
