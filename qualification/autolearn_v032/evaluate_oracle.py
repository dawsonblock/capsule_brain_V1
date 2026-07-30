"""Compute the per-split oracle ceiling (best measured action per task)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.learner import SupervisedRouterLearner

from .config import QualificationConfig
from .evaluate import evaluate_policy_on_split
from .schemas import read_json, write_json


def _oracle_policy_for_split(split: str):
    """A trivial oracle policy is no policy at all; evaluate handles it."""
    # We reuse the baseline object for the select signature but set oracle=True.
    return BaselinePolicyV3()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute oracle ceiling for each split.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--split", choices=["experience", "validation", "test", "ood", "safety"], default="test")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    policy = _oracle_policy_for_split(args.split)
    result = evaluate_policy_on_split(config, args.split, policy, oracle=True, label="oracle")
    path = Path(args.artifacts_dir) / f"evaluate_oracle_{args.split}.json"
    write_json(path, result)
    print(f"oracle {args.split}: mean_utility={result['mean_utility']:.4f} n={result['n_tasks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
