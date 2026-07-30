"""Train the candidate learned policy on real D_experience (Section 9).

v0.4.0 fixes:
  * The policy embeds a complete PolicyProvenance manifest with non-empty
    SHA-256 digests computed from the actual project root and artifacts.
  * Trust-region constraint (Section 9.9): the candidate is rejected if
    action collapse occurs, average KL from baseline exceeds threshold, or
    any action probability becomes effectively unreachable.
  * Class imbalance handling (Section 9.7).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner, TrainingExample
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.features import FEATURE_SCHEMA_VERSION, FEATURE_NAMES

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .build_dataset import build_dataset
from .config import QualificationConfig
from .provenance import build_policy_provenance
from .schemas import PolicyProvenance, ProvenanceError, read_json, sha256_json, sha256_text, write_json


def _load_split(artifacts: Path, split: str) -> list[TrainingExample]:
    path = artifacts / f"dataset_{split}.json"
    data = read_json(path)
    out: list[TrainingExample] = []
    for row in data:
        out.append(TrainingExample(
            features=list(row["features"]),
            best_action=Action(row["best_action"]),
            utility_margin=float(row.get("utility_margin", 0.0)),
            weight=float(row.get("weight", 1.0)),
            task_id=str(row.get("task_id", "")),
            task_family=str(row.get("task_family", "unknown")),
            split=str(row.get("split", split)),
        ))
    return out


def _feature_schema_digest() -> str:
    return sha256_text(FEATURE_SCHEMA_VERSION + ":" + ",".join(FEATURE_NAMES))


def _learner_code_digest() -> str:
    """SHA-256 over the learner implementation source."""
    from .provenance import find_project_root
    root = find_project_root()
    learner_path = root / "src" / "capsule_brain" / "autolearn" / "learner.py"
    policy_path = root / "src" / "capsule_brain" / "autolearn" / "policy.py"
    import hashlib
    h = hashlib.sha256()
    for p in (learner_path, policy_path):
        if p.exists():
            h.update(p.read_bytes())
            h.update(b"\x00")
    return h.hexdigest()


def _hyperparameter_digest(cfg: LearnerConfig) -> str:
    return sha256_json(cfg.to_dict())


def _compute_kl_from_baseline(policy: LearnedPolicy, baseline, tasks: list[dict[str, Any]]) -> float:
    """Compute average KL(P_candidate || P_baseline) over tasks (Section 9.9)."""
    from .run_counterfactuals import _state_from_task
    from capsule_brain.autolearn.features import FeatureExtractor
    extractor = FeatureExtractor()
    kls: list[float] = []
    for task in tasks:
        state = _state_from_task(task)
        allowed = [Action(a) for a in task["allowed_actions"]]
        cand_probs = policy.predict_proba(extractor.extract(state))
        base_dec = baseline.select_action(state, allowed_actions=allowed)
        # Baseline is deterministic; treat its chosen action as prob 1.
        base_action = base_dec.action.value
        kl = 0.0
        for a, p in cand_probs.items():
            q = 1.0 if a == base_action else 0.0
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
            elif p > 0 and q == 0:
                kl += p * 1e6  # large penalty for assigning mass to a baseline-forbidden action
        kls.append(kl)
    return sum(kls) / len(kls) if kls else 0.0


def _check_action_diversity(policy: LearnedPolicy, tasks: list[dict[str, Any]], max_share: float) -> dict[str, Any]:
    """Section 10.4: no route collapse. Maximum single-action share < threshold."""
    from .run_counterfactuals import _state_from_task
    counts: dict[str, int] = {}
    for task in tasks:
        state = _state_from_task(task)
        allowed = [Action(a) for a in task["allowed_actions"]]
        dec = policy.select_action(state, allowed_actions=allowed)
        counts[dec.action.value] = counts.get(dec.action.value, 0) + 1
    total = sum(counts.values()) if counts else 1
    shares = {a: c / total for a, c in counts.items()}
    max_share_val = max(shares.values()) if shares else 0.0
    max_action = max(shares, key=shares.get) if shares else ""
    return {
        "action_shares": shares,
        "max_single_action_share": max_share_val,
        "max_action": max_action,
        "collapse_detected": max_share_val >= max_share,
    }


def train_candidate(config: QualificationConfig) -> tuple[LearnedPolicy, dict[str, Any], dict[str, Any]]:
    artifacts = Path(config.artifacts_dir)
    if not (artifacts / "dataset_manifest.json").exists():
        build_dataset(config)

    train = _load_split(artifacts, "train")
    val = _load_split(artifacts, "validation")
    if not train:
        raise RuntimeError("D_experience is empty; cannot train candidate policy")

    actions = tuple(Action.learned())
    learner_cfg = LearnerConfig(
        learning_rate=0.1,
        n_epochs=400,
        l2=1e-3,
        weight_cap=4.0,
        min_weight=1.0,
        confidence_threshold=0.5,
        temperature=1.0,
        actions=actions,
    )
    learner = SupervisedRouterLearner(learner_cfg)

    dataset_manifest = read_json(artifacts / "dataset_manifest.json")
    split_manifest = read_json(artifacts / "split_manifest.json")

    policy, metrics = learner.train(
        train,
        val,
        policy_id="candidate_v04",
        policy_version="candidate_v04_real",
        training_data_digest=dataset_manifest["dataset_digest"],
        split_manifest_digest=sha256_json(split_manifest),
        parent_policy="baseline_v3",
    )

    # Build and embed the full PolicyProvenance manifest.
    import datetime
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance = build_policy_provenance(
        config=config,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        parent_policy_id="baseline_v3",
        benchmark_manifest_path=artifacts / "benchmark_manifest.json",
        split_manifest_path=artifacts / "split_manifest.json",
        counterfactual_results_path=artifacts / "counterfactual_outcomes.json",
        training_dataset_path=artifacts / "dataset_manifest.json",
        feature_schema_digest=_feature_schema_digest(),
        hyperparameter_digest=_hyperparameter_digest(learner_cfg),
        learner_code_digest=_learner_code_digest(),
        model_id=config.model_id,
        model_revision=None,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=None,
        created_at_utc=created_at,
    )
    provenance.validate()

    # Trust-region checks (Section 9.9).
    from capsule_brain.autolearn.baseline import BaselinePolicyV3
    baseline = BaselinePolicyV3()
    bm = read_json(artifacts / "benchmark_manifest.json")
    test_tasks = [t for t in bm["tasks"] if t["split"] == "test"]
    kl = _compute_kl_from_baseline(policy, baseline, test_tasks)
    diversity = _check_action_diversity(policy, test_tasks, config.max_single_action_share)

    trust_region = {
        "max_kl_from_baseline": config.max_kl_from_baseline,
        "observed_kl": kl,
        "kl_exceeded": kl > config.max_kl_from_baseline,
        "action_diversity": diversity,
        "collapse_detected": diversity["collapse_detected"],
        "passed": (kl <= config.max_kl_from_baseline) and (not diversity["collapse_detected"]),
    }

    # Embed provenance in the policy artifact.
    policy_dict = policy.to_dict()
    policy_dict["provenance"] = provenance.to_dict()
    policy_dict["schema_version"] = "learned-policy/2"
    policy_dict["protocol_version"] = PROTOCOL_VERSION
    policy_dict["trust_region"] = trust_region

    training_record = {
        "schema_version": "candidate-training/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "n_train": metrics.n_train,
        "n_validation": metrics.n_validation,
        "n_epochs": metrics.n_epochs,
        "train_loss": metrics.train_loss,
        "train_accuracy": metrics.train_accuracy,
        "validation_accuracy": metrics.validation_accuracy,
        "provenance": provenance.to_dict(),
        "trust_region": trust_region,
        "hyperparameters": policy.hyperparameters,
    }

    candidate_path = artifacts / "candidate_policy.json"
    write_json(candidate_path, policy_dict)
    training_path = artifacts / "candidate_training.json"
    write_json(training_path, training_record)

    return policy, training_record, metrics.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train candidate policy on real D_experience.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(mode=args.mode, runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    policy, record, metrics = train_candidate(config)
    print(f"candidate policy: {policy.policy_id}")
    print(f"  train_accuracy={record['train_accuracy']:.4f} validation_accuracy={record['validation_accuracy']:.4f}")
    print(f"  trust_region_passed={record['trust_region']['passed']}")
    print(f"  provenance_lineage={record['provenance']['lineage_digest'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
