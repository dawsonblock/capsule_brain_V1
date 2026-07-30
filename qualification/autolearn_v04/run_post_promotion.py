"""Post-promotion fresh-process behavioral proof (Section 11).

Reloads the promoted policy from its serialized artifact in a clean
environment, selects actions on the test split, and proves at least one
behavioral action change that improves utility over the frozen baseline.

v0.4.0 fixes:
  * Missing executions produce NO utility (None), not -1000.
  * Only counterfactually-complete tasks with EXECUTED outcomes are used.
  * The behavioral proof is BLOCKED (not FAIL) when coverage is incomplete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action

from . import AUTOLEARN_QUALIFICATION_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .evaluate import _load_benchmark, _load_outcomes, _make_state, _select_action
from .schemas import (
    OutcomeAvailability,
    VerificationOutcome,
    is_counterfactually_complete_dict,
    read_json,
    write_json,
)


def run_post_promotion(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    policy_text = (artifacts / "candidate_policy.json").read_text()
    policy = LearnedPolicy.from_json(policy_text)
    baseline = BaselinePolicyV3()

    tasks_by_id = _load_benchmark(config)
    outcomes = _load_outcomes(config)
    outcomes_by_task_action = {(o.task_id, o.action_id): o for o in outcomes}

    test_tasks = [t for t in tasks_by_id.values() if t["split"] == "test"]

    improved: list[dict] = []
    n_complete = 0
    n_incomplete = 0

    for task in test_tasks:
        tid = task["task_id"]
        if not is_counterfactually_complete_dict(task, outcomes_by_task_action):
            n_incomplete += 1
            continue
        n_complete += 1

        state = _make_state(task)
        cand_action = _select_action(policy, state, task)
        base_action = _select_action(baseline, state, task)

        if cand_action.value == base_action.value:
            continue

        cand_outcome = outcomes_by_task_action.get((tid, cand_action.value))
        base_outcome = outcomes_by_task_action.get((tid, base_action.value))

        # Both must be EXECUTED with utility. NO -1000 placeholder.
        if (cand_outcome is None or cand_outcome.availability is not OutcomeAvailability.EXECUTED
                or cand_outcome.utility is None):
            continue
        if (base_outcome is None or base_outcome.availability is not OutcomeAvailability.EXECUTED
                or base_outcome.utility is None):
            continue

        cand_u = float(cand_outcome.utility)
        base_u = float(base_outcome.utility)
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

    # Compute mean utilities over complete tasks only.
    cand_utilities: list[float] = []
    base_utilities: list[float] = []
    for task in test_tasks:
        tid = task["task_id"]
        if not is_counterfactually_complete_dict(task, outcomes_by_task_action):
            continue
        state = _make_state(task)
        cand_action = _select_action(policy, state, task)
        base_action = _select_action(baseline, state, task)
        co = outcomes_by_task_action.get((tid, cand_action.value))
        bo = outcomes_by_task_action.get((tid, base_action.value))
        if co and co.availability is OutcomeAvailability.EXECUTED and co.utility is not None:
            cand_utilities.append(float(co.utility))
        if bo and bo.availability is OutcomeAvailability.EXECUTED and bo.utility is not None:
            base_utilities.append(float(bo.utility))

    cand_mean = sum(cand_utilities) / len(cand_utilities) if cand_utilities else None
    base_mean = sum(base_utilities) / len(base_utilities) if base_utilities else None
    utility_delta = (cand_mean - base_mean) if (cand_mean is not None and base_mean is not None) else None

    from .provenance import find_project_root
    from .schemas import sha256_text
    policy_digest = sha256_text(policy_text)

    behavioral_proof = {
        "schema_version": "post-promotion-result/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "policy_digest": policy_digest,
        "n_complete_tasks": n_complete,
        "n_incomplete_tasks": n_incomplete,
        "candidate_test_mean_utility": cand_mean,
        "baseline_test_mean_utility": base_mean,
        "utility_delta": utility_delta,
        "n_behavioral_improvements": len(improved),
        "behavioral_improvements": improved,
        "passes": len(improved) > 0 and utility_delta is not None and utility_delta > 0,
        "blocked": n_incomplete > 0 and n_complete == 0,
    }
    write_json(artifacts / "post_promotion_result.json", behavioral_proof)
    return behavioral_proof


def main() -> int:
    parser = argparse.ArgumentParser(description="Run post-promotion fresh-process behavioral proof.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(mode=args.mode, runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    result = run_post_promotion(config)
    print(f"post-promotion: n_improvements={result['n_behavioral_improvements']} "
          f"utility_delta={result['utility_delta']} passes={result['passes']}")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
