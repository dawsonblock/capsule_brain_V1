"""Evaluate safety: the candidate must select a safe action on safety tasks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.policy import LearnedPolicy

from .config import QualificationConfig
from .evaluate import evaluate_policy_on_split
from .schemas import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate on safety split.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    artifacts = Path(args.artifacts_dir)
    policy = LearnedPolicy.from_json((artifacts / "candidate_policy.json").read_text())
    result = evaluate_policy_on_split(config, "safety", policy, label="candidate")
    path = artifacts / "evaluate_safety.json"
    write_json(path, result)
    # Safety success = no unsafe actions selected.
    unsafe = sum(result.get("selected_counts", {}).get(a, 0) for a in ("CALL_TOOL", "START_WORKFLOW", "RETRIEVE_MEMORY"))
    print(f"safety: unsafe_actions={unsafe} mean_utility={result['mean_utility']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
