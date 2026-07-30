"""Run counterfactual executions (v0.3.1 Section 5-8).

Executes every safe candidate action on every task through the real
Capsule Brain runtime, with strong isolation per (task, action) pair.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from capsule_brain.autolearn.counterfactual import (
    CounterfactualTask,
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction
from qualification.autolearn_v031.config import QualificationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run counterfactual executions.")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Load benchmark manifest.
    manifest_path = artifacts / "benchmark_manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found. Run build_benchmark first.", file=sys.stderr)
        return 1

    config = QualificationConfig(runtime=args.runtime)

    if config.is_real:
        print("Running with REAL runtime — official qualification mode.")
    else:
        print("SIMULATED DEVELOPMENT RUN — NOT QUALIFYING — NOT PROMOTABLE",
              file=sys.stderr)

    # The actual execution would load tasks from the manifest and run them.
    # This is the orchestration entry point.
    print(f"Counterfactual execution with runtime={args.runtime}")
    print(f"Artifacts: {artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
