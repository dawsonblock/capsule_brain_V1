"""AutoLearn v0.3.1 pipeline stage: build_dataset.

This module is part of the single-command qualification pipeline.
See run_all.py for the full pipeline orchestration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run build_dataset stage.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    print(f"Stage: build_dataset")
    print(f"Runtime: {args.runtime}")
    print(f"Artifacts: {artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
