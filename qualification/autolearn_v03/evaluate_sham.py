"""Evaluate the sham policy on held-out tasks (v2.16.0 Priority 4).

Evaluates the sham-learning control policy on the same TEST and OOD tasks
as the real evaluation. The sham policy uses identical architecture but
destroyed experience signal.

Writes sham_evaluation.json with the same structure as evaluation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.counterfactual import (
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import build_all_tasks, build_split_assignments
from evaluate_policy import _evaluate_split, _build_runtime


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sham policy (v2.16.0)")
    parser.add_argument(
        "--runtime",
        choices=["real", "simulated"],
        default="real",
        help="Evaluation runtime: 'real' (default) or 'simulated'.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    sham_id_path = out_dir / "sham_policy_id.txt"
    if not sham_id_path.exists():
        print("ERROR: sham_policy_id.txt not found. Run train_sham.py first.")
        sys.exit(1)

    sham_id = sham_id_path.read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    sham_policy = registry.load_policy(sham_id)

    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)
    test_tasks = [t for t in tasks if assignments.get(t.task_id) == "test"]
    ood_tasks = [t for t in tasks if assignments.get(t.task_id) == "ood"]

    print(f"test tasks: {len(test_tasks)}, ood tasks: {len(ood_tasks)}")

    extractor = FeatureExtractor()
    baseline = BaselinePolicyV2()
    runtime = _build_runtime(args.runtime)

    if args.runtime == "real":
        try:
            assert_runtime_is_real(runtime)
        except RuntimeError:
            print("ERROR: official qualification requires runtime_type='real'")
            sys.exit(1)

    utility_config = UtilityConfig()

    print(f"runtime_type: {runtime.runtime_type}")
    test_eval = _evaluate_split(test_tasks, baseline, sham_policy, extractor, runtime, utility_config)
    ood_eval = _evaluate_split(ood_tasks, baseline, sham_policy, extractor, runtime, utility_config)

    output = {
        "sham_policy_id": sham_policy.policy_id,
        "runtime_type": runtime.runtime_type,
        "test": test_eval,
        "ood": ood_eval,
    }
    text = json.dumps(output, sort_keys=True, indent=2)
    (out_dir / "sham_evaluation.json").write_text(text)
    print(f"wrote {out_dir / 'sham_evaluation.json'}")
    print(f"sham test mean_delta: {test_eval['mean_delta']:.4f}")
    print(f"sham ood mean_delta: {ood_eval['mean_delta']:.4f}")


if __name__ == "__main__":
    main()
