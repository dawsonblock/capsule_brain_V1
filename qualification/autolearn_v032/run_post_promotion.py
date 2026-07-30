"""Post-promotion fresh-process behavioral proof (Section 11).

Reloads the promoted policy from its serialized artifact in a clean
environment, selects actions on the test split, and proves at least one
behavioral action change that improves utility over the frozen baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.source_hash import compute_string_hash

from .config import QualificationConfig
from .evaluate import evaluate_policy_on_split, _load_benchmark, _load_rows
from .schemas import read_json, write_json


def run_post_promotion(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    # Fresh reload: read policy from disk, do not reuse in-memory object.
    policy_text = (artifacts / "candidate_policy.json").read_text()
    policy = LearnedPolicy.from_json(policy_text)
    baseline = BaselinePolicyV3()

    test_res = evaluate_policy_on_split(config, "test", policy, baseline=baseline, label="candidate")

    # Find at least one behavioral change that improves utility.
    tasks_by_id = _load_benchmark(config)
    rows_by_task = _load_rows(config)
    improved: list[dict] = []
    for task in tasks_by_id.values():
        if task["split"] != "test":
            continue
        tid = task["task_id"]
        task_rows = rows_by_task.get(tid, [])
        if not task_rows:
            continue
        from .run_counterfactuals import _state_from_task
        state = _state_from_task(task)
        cand_action = policy.select_action(state, allowed_actions=[Action(a) for a in task["allowed_actions"]]).action
        base_action = baseline.select_action(state, allowed_actions=[Action(a) for a in task["allowed_actions"]]).action
        if cand_action == base_action:
            continue
        cand_u = next((r["utility"] for r in task_rows if r["action"] == cand_action.value), -1000.0)
        base_u = next((r["utility"] for r in task_rows if r["action"] == base_action.value), -1000.0)
        if cand_u > base_u:
            improved.append({
                "task_id": tid,
                "family": task["family"],
                "baseline_action": base_action.value,
                "candidate_action": cand_action.value,
                "baseline_utility": base_u,
                "candidate_utility": cand_u,
                "improvement": cand_u - base_u,
            })

    behavioral_proof = {
        "package_version": "2.15.3",
        "autolearn_qualification_version": "0.3.2",
        "policy_digest": compute_string_hash(policy_text),
        "test_mean_utility": test_res["mean_utility"],
        "baseline_test_mean_utility": test_res.get("vs_baseline", {}).get("mean_delta", 0) - test_res["mean_utility"]
        if test_res.get("vs_baseline") else 0,
        "n_behavioral_improvements": len(improved),
        "behavioral_improvements": improved,
        "passes": len(improved) > 0,
    }

    # The baseline_test_mean_utility should be computed; the above formula is
    # only a placeholder. Recompute from baseline directly.
    baseline_res = evaluate_policy_on_split(config, "test", baseline, label="baseline")
    behavioral_proof["baseline_test_mean_utility"] = baseline_res["mean_utility"]
    behavioral_proof["utility_delta"] = test_res["mean_utility"] - baseline_res["mean_utility"]

    path = artifacts / "post_promotion_result.json"
    write_json(path, behavioral_proof)
    return behavioral_proof


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-promotion fresh-process behavioral proof.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    result = run_post_promotion(config)
    print(f"post-promotion: n_improvements={result['n_behavioral_improvements']} "
          f"utility_delta={result['utility_delta']:.4f} passes={result['passes']}")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
