"""Common evaluation logic for validation, test, OOD, oracle, and safety (Section 10/11).

Evaluates a policy by replaying the real measured counterfactual utilities:
for each task in the split, the policy selects an action; the selected
action's measured utility is the task's utility. No live re-execution during
eval — the outcome is the real measured one.

Includes paired bootstrap BCa, sign test, per-family precision/recall, and
utility deltas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState
from capsule_brain.autolearn.statistics import paired_bootstrap

from .config import QualificationConfig
from .run_counterfactuals import _state_from_task
from .schemas import read_json, write_json


def _action_allowed_for_task(action: Action, task: dict) -> bool:
    return action.value in task["allowed_actions"]


def _select_action(policy, state: ExecutiveState, task: dict) -> Action:
    allowed = [Action(a) for a in task["allowed_actions"]]
    decision = policy.select_action(state, allowed_actions=allowed)
    return decision.action


def _rows_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in rows:
        d.setdefault(r["task_id"], []).append(r)
    return d


def _make_state(task: dict) -> ExecutiveState:
    return _state_from_task(task)


def _load_rows(config: QualificationConfig) -> dict[str, list[dict]]:
    artifacts = Path(config.artifacts_dir)
    rows = read_json(artifacts / "real_counterfactual_results.json")
    return _rows_by_task(rows)


def _load_benchmark(config: QualificationConfig) -> dict[str, dict]:
    artifacts = Path(config.artifacts_dir)
    bm = read_json(artifacts / "benchmark_manifest.json")
    return {t["task_id"]: t for t in bm["tasks"]}


def evaluate_policy_on_split(
    config: QualificationConfig,
    split: str,
    policy,
    *,
    baseline=None,
    oracle: bool = False,
    label: str = "policy",
) -> dict[str, Any]:
    """Evaluate a policy on the named split using real measured utilities.

    Args:
        policy: policy with select_action(state, allowed_actions=...)
        split: one of experience/validation/test/ood/safety
        oracle: if True, the oracle is the argmax of measured utilities per task
        baseline: optional baseline for delta computation
    """
    artifacts = Path(config.artifacts_dir)
    tasks_by_id = _load_benchmark(config)
    rows_by_task = _load_rows(config)

    split_tasks = [t for t in tasks_by_id.values() if t["split"] == split]
    if not split_tasks:
        return {
            "split": split,
            "policy": label,
            "n_tasks": 0,
            "mean_utility": 0.0,
            "pass": False,
            "status": "NO_DATA",
            "message": f"no {split} tasks in counterfactual results",
        }

    utilities: list[float] = []
    successes: list[int] = []
    per_family: dict[str, Any] = {}
    selected_counts: dict[str, int] = {}
    oracle_utilities: list[float] = []
    deltas: list[float] = []
    task_results: list[dict] = []

    for task in split_tasks:
        tid = task["task_id"]
        task_rows = rows_by_task.get(tid, [])
        state = _make_state(task)
        action = _select_action(policy, state, task)

        if oracle:
            # Oracle: pick the measured argmax utility action.
            best_row = max(task_rows, key=lambda r: r["utility"], default=None)
            if best_row:
                action = Action(best_row["action"])

        matching = [r for r in task_rows if r["action"] == action.value]
        if not matching:
            # No measured row: treat as not executed (large negative).
            u = -1000.0
            success = False
        else:
            row = matching[0]
            u = float(row["utility"])
            success = bool(row["verified_success"])

        utilities.append(u)
        successes.append(1 if success else 0)
        selected_counts[action.value] = selected_counts.get(action.value, 0) + 1

        family = task["family"]
        if family not in per_family:
            per_family[family] = {
                "n": 0,
                "mean_utility": 0.0,
                "success_rate": 0.0,
                "selected": {},
            }
        per_family[family]["n"] += 1
        per_family[family]["selected"][action.value] = per_family[family]["selected"].get(action.value, 0) + 1

        # Oracle utility for gap.
        if task_rows:
            oracle_u = max(r["utility"] for r in task_rows)
            oracle_utilities.append(oracle_u)
        else:
            oracle_utilities.append(-1000.0)

        if baseline is not None:
            base_action = _select_action(baseline, state, task)
            base_matching = [r for r in task_rows if r["action"] == base_action.value]
            if not base_matching:
                base_u = -1000.0
            else:
                base_u = float(base_matching[0]["utility"])
            deltas.append(u - base_u)

        task_results.append({
            "task_id": tid,
            "family": family,
            "selected_action": action.value,
            "utility": u,
            "verified_success": success,
        })

    for fam, stats in per_family.items():
        fam_rows = [r for r in task_results if r["family"] == fam]
        if fam_rows:
            stats["mean_utility"] = sum(r["utility"] for r in fam_rows) / len(fam_rows)
            stats["success_rate"] = sum(1 for r in fam_rows if r["verified_success"]) / len(fam_rows)

    mean_u = sum(utilities) / len(utilities)
    success_rate = sum(successes) / len(successes)
    oracle_mean = sum(oracle_utilities) / len(oracle_utilities)
    gap = oracle_mean - mean_u

    result: dict[str, Any] = {
        "split": split,
        "policy": label,
        "n_tasks": len(split_tasks),
        "mean_utility": mean_u,
        "success_rate": success_rate,
        "oracle_mean_utility": oracle_mean,
        "oracle_gap": gap,
        "selected_counts": selected_counts,
        "per_family": per_family,
        "task_results": task_results,
    }

    if deltas:
        b = paired_bootstrap(deltas, n_bootstrap=config.bootstrap_replicates, seed=config.bootstrap_seed)
        result["vs_baseline"] = {
            "mean_delta": b.mean,
            "ci_low": b.ci_low,
            "ci_high": b.ci_high,
            "n_samples": b.n_samples,
            "passes_threshold": b.passes_threshold,
            "std_error": b.std_error,
        }

    # Promotion-style coverage/precision/recall for tool/workflow.
    if split in {"test", "ood"}:
        result["tool_metrics"] = _tool_metrics(task_results, rows_by_task, split_tasks)
        result["workflow_metrics"] = _workflow_metrics(task_results, rows_by_task, split_tasks)

    return result


def _family_is_required(family: str, action: str) -> bool:
    return (family == "tool_required" and action == "CALL_TOOL") or \
           (family == "workflow_required" and action == "START_WORKFLOW") or \
           (family == "memory_required" and action == "RETRIEVE_MEMORY") or \
           (family == "direct_answer" and action == "ANSWER_DIRECT")


def _tool_metrics(task_results, rows_by_task, split_tasks):
    tp = fp = fn = 0
    for t in split_tasks:
        tid = t["task_id"]
        row = [r for r in task_results if r["task_id"] == tid][0]
        is_tool = t["family"] == "tool_required"
        sel = row["selected_action"]
        if is_tool and sel == "CALL_TOOL":
            tp += 1
        elif not is_tool and sel == "CALL_TOOL":
            fp += 1
        elif is_tool and sel != "CALL_TOOL":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _workflow_metrics(task_results, rows_by_task, split_tasks):
    tp = fp = fn = 0
    for t in split_tasks:
        tid = t["task_id"]
        row = [r for r in task_results if r["task_id"] == tid][0]
        is_wf = t["family"] == "workflow_required"
        sel = row["selected_action"]
        if is_wf and sel == "START_WORKFLOW":
            tp += 1
        elif not is_wf and sel == "START_WORKFLOW":
            fp += 1
        elif is_wf and sel != "START_WORKFLOW":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def run_evaluation(config: QualificationConfig, split: str, label: str = "candidate") -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    policy = LearnedPolicy.from_json((artifacts / "candidate_policy.json").read_text())
    baseline = BaselinePolicyV3()
    result = evaluate_policy_on_split(config, split, policy, baseline=baseline, label=label)
    path = artifacts / f"evaluate_{split}.json"
    write_json(path, result)
    return result


def main_for_split(split: str) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=f"Evaluate candidate on {split} split.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    result = run_evaluation(config, split)
    print(f"{split}: mean_utility={result['mean_utility']:.4f} success_rate={result.get('success_rate', 0):.4f}")
    return 0
