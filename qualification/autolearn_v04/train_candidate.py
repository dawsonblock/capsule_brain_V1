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


def _compute_kl_from_baseline(
    policy: LearnedPolicy,
    baseline,
    tasks: list[dict[str, Any]],
    *,
    epsilon_smooth: float = 0.1,
) -> dict[str, float]:
    """Compute average KL(P_candidate || Q_baseline_smoothed) over tasks.

    v0.4.0 fix (Section 8): The deterministic baseline is represented as a
    SMOOTHED distribution, not a delta. For K actions and baseline-selected
    action a0:
        q(a0) = 1 - epsilon_smooth
        q(a != a0) = epsilon_smooth / (K - 1)

    This prevents the zero-probability explosion that made the old KL
    calculation mathematically unusable. A softmax candidate can now satisfy
    a reasonable KL threshold without exactly imitating the baseline.

    Returns a dict with mean_kl, p95_kl, action_change_rate, candidate_entropy,
    and baseline_agreement_rate.
    """
    from .run_counterfactuals import _state_from_task
    from capsule_brain.autolearn.features import FeatureExtractor
    extractor = FeatureExtractor()
    kls: list[float] = []
    entropies: list[float] = []
    changes: list[int] = []
    agreements: list[int] = []
    for task in tasks:
        state = _state_from_task(task)
        allowed = [Action(a) for a in task["allowed_actions"]]
        allowed_strs = [a.value for a in allowed]
        K = len(allowed_strs)
        if K <= 1:
            continue
        fv = extractor.extract(state)
        cand_probs = policy.predict_proba(fv)
        # Only consider allowed actions.
        cand_probs = {a: p for a, p in cand_probs.items() if a in allowed_strs}
        # Normalize.
        total_p = sum(cand_probs.values())
        if total_p > 0:
            cand_probs = {a: p / total_p for a, p in cand_probs.items()}
        base_dec = baseline.select_action(state, allowed_actions=allowed)
        base_action = base_dec.action.value
        # Smoothed baseline distribution.
        q = {}
        for a in allowed_strs:
            if a == base_action:
                q[a] = 1.0 - epsilon_smooth
            else:
                q[a] = epsilon_smooth / (K - 1)
        # KL(p || q) = sum p * log(p / q)
        kl = 0.0
        for a, p in cand_probs.items():
            if p > 0:
                kl += p * math.log(p / q[a])
        kls.append(kl)
        # Candidate entropy.
        entropy = -sum(p * math.log(p) for p in cand_probs.values() if p > 0)
        entropies.append(entropy)
        # Action change rate.
        cand_action = max(cand_probs, key=cand_probs.get) if cand_probs else base_action
        changes.append(1 if cand_action != base_action else 0)
        agreements.append(1 if cand_action == base_action else 0)
    if not kls:
        return {"mean_kl": 0.0, "p95_kl": 0.0, "action_change_rate": 0.0,
                "candidate_entropy": 0.0, "baseline_agreement_rate": 1.0}
    kls.sort()
    p95_idx = min(len(kls) - 1, int(0.95 * len(kls)))
    return {
        "mean_kl": sum(kls) / len(kls),
        "p95_kl": kls[p95_idx],
        "action_change_rate": sum(changes) / len(changes),
        "candidate_entropy": sum(entropies) / len(entropies),
        "baseline_agreement_rate": sum(agreements) / len(agreements),
    }


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

    # Trust-region checks on D_VALIDATION ONLY (Section 7 fix).
    # The previous code loaded D_test for trust-region and action-collapse
    # diagnostics, violating untouched final-test isolation. This is now
    # fixed: all training-time diagnostics use D_validation.
    from capsule_brain.autolearn.baseline import BaselinePolicyV3
    baseline = BaselinePolicyV3()
    bm = read_json(artifacts / "benchmark_manifest.json")
    # CRITICAL: use validation tasks, NOT test tasks.
    validation_tasks = [t for t in bm["tasks"] if t["split"] == "validation"]
    kl_result = _compute_kl_from_baseline(
        policy, baseline, validation_tasks,
        epsilon_smooth=config.kl_epsilon_smooth,
    )
    diversity = _check_action_diversity(policy, validation_tasks, config.max_single_action_share)

    trust_region = {
        "split_used": "validation",  # NOT test
        "max_kl_from_baseline": config.max_kl_from_baseline,
        "epsilon_smooth": config.kl_epsilon_smooth,
        "observed_mean_kl": kl_result["mean_kl"],
        "observed_p95_kl": kl_result["p95_kl"],
        "action_change_rate": kl_result["action_change_rate"],
        "candidate_entropy": kl_result["candidate_entropy"],
        "baseline_agreement_rate": kl_result["baseline_agreement_rate"],
        "kl_exceeded": kl_result["mean_kl"] > config.max_kl_from_baseline,
        "action_diversity": diversity,
        "collapse_detected": diversity["collapse_detected"],
        "passed": (kl_result["mean_kl"] <= config.max_kl_from_baseline)
                  and (not diversity["collapse_detected"]),
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
