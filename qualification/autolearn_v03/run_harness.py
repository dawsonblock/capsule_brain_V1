"""Full learning-cycle qualification harness for AutoLearn v0.3.

Section 26: This script runs the entire qualification pipeline:
1. Generate benchmark manifest
2. Run counterfactuals (real or simulated with --allow-simulated)
3. Build dataset
4. Train policy
5. Evaluate policy (test + OOD)
6. Run shadow eval
7. Promote policy (12-gate)
8. Generate remaining artifacts
9. Generate provenance manifest
10. Generate report

Usage:
    python run_harness.py --allow-simulated   # for testing
    python run_harness.py                      # for official qualification (requires real runtime)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, extra_args: list[str] | None = None) -> int:
    """Run a Python script in this directory."""
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        return result.returncode
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full AutoLearn v0.3 qualification harness")
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help="Allow the simulated runtime (for testing only)",
    )
    args = parser.parse_args()

    cf_args = ["--allow-simulated"] if args.allow_simulated else []

    steps = [
        ("benchmark_manifest.py", []),
        ("run_counterfactuals.py", cf_args),
        ("build_dataset.py", []),
        ("train_policy.py", []),
        ("evaluate_policy.py", []),
        ("run_shadow_eval.py", []),
        ("promote_policy.py", []),
        ("generate_artifacts.py", []),
        ("provenance_manifest.py", []),
        ("report.py", []),
    ]

    for script, extra in steps:
        rc = _run_script(script, extra)
        if rc != 0:
            print(f"\nHarness FAILED at step: {script}")
            sys.exit(rc)

    print(f"\n{'='*60}")
    print("Harness completed successfully!")
    print(f"{'='*60}")
    print(f"Report: {Path(__file__).resolve().parent / 'AUTOLEARN_V0_3_REPORT.md'}")


if __name__ == "__main__":
    main()
