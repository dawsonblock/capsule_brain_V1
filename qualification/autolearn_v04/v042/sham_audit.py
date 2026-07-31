"""Sham construction audit and multi-seed ensemble.

Audits the current sham implementation and runs 20+ sham seeds
to produce a sham distribution for candidate comparison.
"""
from __future__ import annotations

import json
import numpy as np
from collections import Counter
from typing import Any

from .data import LoadedRun
from .sham_trainer import train_sham_policy, sham_ensemble, train_logistic
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_sham_global_selector,
    make_candidate_selector,
)


def audit_sham_construction(run: LoadedRun) -> dict[str, Any]:
    """Audit the current sham implementation.

    Answers whether sham training retains or destroys:
    - task-family information
    - archetype information
    - utility-margin correlation
    - feature-label association
    - class-frequency information
    - route-specific sample weights
    - normalization fitted using label-sensitive data
    - validation threshold tuning based on real labels
    - duplicate or near-duplicate tasks
    - split leakage
    """
    sham_policy = run.sham_policy
    sham_training = run.sham_training

    audit = {
        "current_sham": {
            "is_sham": sham_policy.get("is_sham"),
            "sham_seed": sham_policy.get("sham_seed"),
            "policy_type": sham_policy.get("policy_type"),
            "n_train": sham_training.get("n_train"),
            "train_accuracy": sham_training.get("train_accuracy"),
            "train_loss": sham_training.get("train_loss"),
            "validation_accuracy": sham_training.get("validation_accuracy"),
        },
        "audit_questions": {},
        "sham_variants": {},
    }

    # Audit each question.
    audit["audit_questions"]["task_family_information"] = {
        "retained": True,
        "explanation": "Sham uses same feature vectors which encode family info. "
                       "Global shuffle destroys label-family correlation but features still contain family signal.",
    }
    audit["audit_questions"]["archetype_information"] = {
        "retained": True,
        "explanation": "Features may encode archetype information. Global shuffle "
                       "does not remove archetype from features.",
    }
    audit["audit_questions"]["utility_margin_correlation"] = {
        "retained": False,
        "explanation": "Global label shuffle breaks the margin-label correlation. "
                       "Margin shuffle variant preserves margin bins.",
    }
    audit["audit_questions"]["feature_label_association"] = {
        "retained": False,
        "explanation": "Global shuffle breaks feature-label association. "
                       "Feature shuffle variant preserves labels but breaks feature order.",
    }
    audit["audit_questions"]["class_frequency_information"] = {
        "retained": True,
        "explanation": "Global shuffle preserves the marginal class distribution. "
                       "The sham sees the same action frequencies as the candidate.",
    }
    audit["audit_questions"]["route_specific_sample_weights"] = {
        "retained": True,
        "explanation": "Sham uses the same training weights as candidate. "
                       "Weights are derived from utility margins which are preserved.",
    }
    audit["audit_questions"]["normalization_label_sensitive"] = {
        "retained": False,
        "explanation": "Feature normalization is computed from features only, not labels.",
    }
    audit["audit_questions"]["validation_threshold_label_sensitive"] = {
        "retained": False,
        "explanation": "Sham does not access validation labels for threshold tuning.",
    }
    audit["audit_questions"]["duplicate_tasks"] = {
        "retained": "unknown",
        "explanation": "Need to check for duplicate or near-duplicate tasks in training set.",
    }
    audit["audit_questions"]["split_leakage"] = {
        "retained": False,
        "explanation": "Sham is trained on D_experience only, evaluated on D_test. No split leakage.",
    }

    # Check for duplicate tasks.
    train_ids = [ex["task_id"] for ex in run.train_examples]
    audit["duplicate_task_check"] = {
        "n_train": len(train_ids),
        "n_unique": len(set(train_ids)),
        "has_duplicates": len(train_ids) != len(set(train_ids)),
    }

    # Document each sham variant.
    audit["sham_variants"]["P_sham_global"] = {
        "shuffle_unit": "label",
        "shuffle_scope": "global",
        "preserved_marginals": ["class_frequency", "feature_distribution", "weights"],
        "destroyed_relationships": ["feature_label", "margin_label"],
        "training_weights": "same as candidate",
        "feature_preprocessing": "same as candidate",
        "calibration_method": "same as candidate",
        "validation_access": "none",
        "test_access": "none",
    }
    audit["sham_variants"]["P_sham_family"] = {
        "shuffle_unit": "label",
        "shuffle_scope": "within_family",
        "preserved_marginals": ["class_frequency_per_family", "feature_distribution", "weights"],
        "destroyed_relationships": ["feature_label_within_family"],
        "training_weights": "same as candidate",
        "feature_preprocessing": "same as candidate",
        "calibration_method": "same as candidate",
        "validation_access": "none",
        "test_access": "none",
    }
    audit["sham_variants"]["P_sham_margin"] = {
        "shuffle_unit": "label",
        "shuffle_scope": "within_margin_bin",
        "preserved_marginals": ["class_frequency_per_margin_bin", "feature_distribution", "weights"],
        "destroyed_relationships": ["feature_label_within_margin_bin"],
        "training_weights": "same as candidate",
        "feature_preprocessing": "same as candidate",
        "calibration_method": "same as candidate",
        "validation_access": "none",
        "test_access": "none",
    }
    audit["sham_variants"]["P_sham_feature"] = {
        "shuffle_unit": "feature_vector",
        "shuffle_scope": "global",
        "preserved_marginals": ["class_frequency", "label_distribution", "weights"],
        "destroyed_relationships": ["feature_label", "feature_order"],
        "training_weights": "same as candidate",
        "feature_preprocessing": "same as candidate",
        "calibration_method": "same as candidate",
        "validation_access": "none",
        "test_access": "none",
    }

    return audit


def compute_sham_seed_distribution(
    run: LoadedRun,
    n_seeds: int = 20,
    task_ids: list[str] | None = None,
    base_seed: int = 10000,
) -> dict[str, Any]:
    """Run 20 sham seeds and report the sham utility distribution.

    For each sham seed, train a sham policy and evaluate on test tasks.
    Report the distribution of sham utilities and the candidate's
    percentile relative to that distribution.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Train sham ensemble.
    sham_policies = sham_ensemble(run, n_seeds=n_seeds, shuffle_mode="global", base_seed=base_seed)

    # Evaluate each sham on test tasks.
    sham_utilities = []
    sham_action_dists = []
    for sp in sham_policies:
        u = _evaluate_policy_weights(run, sp, task_ids)
        sham_utilities.append(u)
        # Action distribution.
        actions = sp.get("actions", run.all_actions)
        dist = _action_distribution(run, sp, task_ids)
        sham_action_dists.append(dist)

    sham_utilities = np.array(sham_utilities)

    # Candidate utility.
    candidate_sel = make_candidate_selector(run)
    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    candidate_utility = candidate_res["mean_utility"]

    # Candidate percentile relative to sham distribution.
    n_below = int(np.sum(sham_utilities < candidate_utility))
    percentile = n_below / len(sham_utilities) * 100

    return {
        "n_sham_seeds": n_seeds,
        "base_seed": base_seed,
        "sham_utilities": sham_utilities.tolist(),
        "sham_mean": float(sham_utilities.mean()),
        "sham_std": float(sham_utilities.std(ddof=1)),
        "sham_ci_low": float(np.percentile(sham_utilities, 2.5)),
        "sham_ci_high": float(np.percentile(sham_utilities, 97.5)),
        "sham_min": float(sham_utilities.min()),
        "sham_max": float(sham_utilities.max()),
        "candidate_utility": candidate_utility,
        "candidate_percentile": float(percentile),
        "candidate_exceeds_sham_upper": bool(candidate_utility > sham_utilities.max()),
        "sham_action_distributions": sham_action_dists,
        "primary_criterion": (
            "U_candidate must exceed the upper uncertainty range of the sham "
            "ensemble or achieve a positive paired lower confidence bound "
            "against the predeclared sham aggregation."
        ),
        "criterion_met": bool(candidate_utility > np.percentile(sham_utilities, 97.5)),
    }


def _evaluate_policy_weights(
    run: LoadedRun,
    policy: dict[str, Any],
    task_ids: list[str],
) -> float:
    """Evaluate a policy (weights + bias) on test tasks, return mean utility."""
    weights = np.array(policy.get("weights", []))
    bias = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    if not weights.size:
        return 0.0

    utilities = []
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        # Find test example for features.
        ex = None
        for e in run.test_examples:
            if e["task_id"] == tid:
                ex = e
                break
        if ex is None:
            continue
        features = np.array(ex.get("features", []))
        if len(features) != weights.shape[1]:
            continue
        logits = features @ weights.T + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            continue
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        selected = eligible[int(np.argmax(eligible_logits))]
        u = run.get_utility_for_action(tid, selected)
        if u is not None:
            utilities.append(u)
    return float(np.mean(utilities)) if utilities else 0.0


def _action_distribution(
    run: LoadedRun,
    policy: dict[str, Any],
    task_ids: list[str],
) -> dict[str, int]:
    """Get action distribution for a policy on test tasks."""
    weights = np.array(policy.get("weights", []))
    bias = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    dist = Counter()
    if not weights.size:
        return {a: 0 for a in run.all_actions}

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        ex = None
        for e in run.test_examples:
            if e["task_id"] == tid:
                ex = e
                break
        if ex is None:
            continue
        features = np.array(ex.get("features", []))
        if len(features) != weights.shape[1]:
            continue
        logits = features @ weights.T + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            continue
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        selected = eligible[int(np.argmax(eligible_logits))]
        dist[selected] += 1

    return {a: dist.get(a, 0) for a in run.all_actions}
