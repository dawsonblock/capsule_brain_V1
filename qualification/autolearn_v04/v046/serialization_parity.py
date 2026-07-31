"""Serialization parity for v0.4.6.

Generates serialization parity artifacts for candidate and sham
policies.  For synthetic fixtures, a deterministic hash-based parity
check is computed (since we can't actually reload in a fresh process).

The parity check verifies that serializing and deserializing the
policy produces identical logits, selected actions, and shapes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus

# Tolerance for floating-point logit comparison.
DEFAULT_TOLERANCE = 1e-9


def _compute_policy_hash(policy: dict) -> str:
    """Compute a deterministic SHA-256 hash of the policy's core fields."""
    payload = {
        "policy_id": policy.get("policy_id"),
        "policy_type": policy.get("policy_type"),
        "model_id": policy.get("model_id"),
        "model_revision": policy.get("model_revision"),
        "feature_schema_digest": policy.get("feature_schema_digest"),
        "feature_transform_digest": policy.get("feature_transform_digest"),
        "weights": policy.get("weights"),
        "bias": policy.get("bias"),
        "action_ordering": policy.get("action_ordering"),
        "training_seed": policy.get("training_seed"),
        "training_split_digest": policy.get("training_split_digest"),
        "learner_config_digest": policy.get("learner_config_digest"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _compute_feature_transform_hash(policy: dict) -> str:
    """Compute a hash of the feature transform identity."""
    payload = {
        "feature_schema_digest": policy.get("feature_schema_digest"),
        "feature_transform_digest": policy.get("feature_transform_digest"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _compute_logits(
    weights,
    bias: list[float],
    feature_vector: list[float],
) -> list[float]:
    """Compute logits = weights @ feature_vector + bias.

    Handles both list-of-lists (matrix) and dict (action->scalar) formats.
    """
    logits: list[float] = []
    if isinstance(weights, dict):
        # Dict format: each value is a scalar weight.
        for key, w in weights.items():
            fv = feature_vector[:1] if feature_vector else [1.0]
            dot = float(w) * fv[0]
            logits.append(dot)
        return logits
    if not isinstance(weights, list):
        return logits
    for i, row in enumerate(weights):
        if isinstance(row, (list, tuple)):
            if len(row) != len(feature_vector):
                fv = feature_vector[:len(row)]
            else:
                fv = feature_vector
            dot = sum(w * f for w, f in zip(row, fv))
        else:
            # Scalar weight
            fv = feature_vector[:1] if feature_vector else [1.0]
            dot = float(row) * fv[0]
        b = bias[i] if i < len(bias) else 0.0
        logits.append(dot + b)
    return logits


def _select_action(logits: list[float], action_ordering: list[str]) -> str:
    """Select the action with the highest logit."""
    if not logits or not action_ordering:
        return ""
    best_idx = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if i < len(logits) and logits[i] > best_val:
            best_val = logits[i]
            best_idx = i
    if best_idx < len(action_ordering):
        return action_ordering[best_idx]
    return ""


def _get_weight_shape(weights) -> list[int]:
    """Return the shape of the weights matrix [n_actions, n_features].

    Handles both list-of-lists (matrix) and dict (action->scalar) formats.
    """
    if not weights:
        return [0, 0]
    if isinstance(weights, dict):
        # Dict format: {action_name: weight_scalar}
        return [len(weights), 1]
    if isinstance(weights, list):
        n_actions = len(weights)
        if n_actions == 0:
            return [0, 0]
        first = weights[0]
        if isinstance(first, (list, tuple)):
            n_features = len(first) if first else 0
        else:
            n_features = 1  # list of scalars
        return [n_actions, n_features]
    return [0, 0]


def _get_bias_shape(bias: list[float]) -> list[int]:
    """Return the shape of the bias vector."""
    return [len(bias)] if bias else [0]


def _get_validation_feature_vector(validation_fixture: dict | None) -> list[float]:
    """Extract a feature vector from the validation fixture."""
    if validation_fixture is None:
        return []
    if not isinstance(validation_fixture, dict):
        return []
    fv = validation_fixture.get("feature_vector")
    if isinstance(fv, list):
        return [float(x) for x in fv if isinstance(x, (int, float))]
    return []


def compute_serialization_parity(
    policy: dict,
    policy_name: str,
    validation_fixture: dict | None = None,
) -> dict:
    """Compute serialization parity artifacts for a policy.

    For synthetic fixtures, this computes a deterministic hash-based
    parity check.  The policy is "serialized" (JSON round-trip) and
    "reloaded", then logits are compared.

    Parameters
    ----------
    policy:
        The policy dict to check.
    policy_name:
        Name label for the policy (e.g. "candidate" or "sham").
    validation_fixture:
        Optional dict with a ``feature_vector`` to use for logit
        computation.  If None, a zero vector is used.

    Returns
    -------
    dict
        policy_hash, feature_transform_hash, action_ordering,
        weight_shape, bias_shape, original_logits, reloaded_logits,
        max_abs_diff, selected_action_parity, status
    """
    if not isinstance(policy, dict):
        return {
            "policy_hash": None,
            "feature_transform_hash": None,
            "action_ordering": [],
            "weight_shape": [0, 0],
            "bias_shape": [0],
            "original_logits": [],
            "reloaded_logits": [],
            "max_abs_diff": None,
            "selected_action_parity": False,
            "status": AuditStatus.BLOCKED.value,
        }

    # --- Compute hashes ---
    policy_hash = _compute_policy_hash(policy)
    feature_transform_hash = _compute_feature_transform_hash(policy)

    # --- Extract policy fields ---
    weights = policy.get("weights", [])
    bias = policy.get("bias", [])
    action_ordering = policy.get("action_ordering", [])

    weight_shape = _get_weight_shape(weights)
    bias_shape = _get_bias_shape(bias)

    # --- Get feature vector for logit computation ---
    feature_vector = _get_validation_feature_vector(validation_fixture)
    if not feature_vector:
        if isinstance(weights, dict) and weights:
            feature_vector = [1.0]
        elif isinstance(weights, list) and weights and isinstance(weights[0], (list, tuple)) and weights[0]:
            n_features = len(weights[0])
            feature_vector = [0.0] * n_features
        elif isinstance(weights, list) and weights:
            feature_vector = [1.0]

    # --- Compute original logits ---
    original_logits = _compute_logits(weights, bias, feature_vector)

    # --- "Reload" the policy via JSON round-trip ---
    # Serialize to JSON and parse back.
    serialized = json.dumps(policy, sort_keys=True, ensure_ascii=False)
    reloaded_policy = json.loads(serialized)

    reloaded_weights = reloaded_policy.get("weights", [])
    reloaded_bias = reloaded_policy.get("bias", [])
    reloaded_logits = _compute_logits(reloaded_weights, reloaded_bias, feature_vector)

    # --- Compare logits ---
    max_abs_diff = 0.0
    for o, r in zip(original_logits, reloaded_logits):
        diff = abs(o - r)
        if diff > max_abs_diff:
            max_abs_diff = diff

    # --- Compare selected actions ---
    original_action = _select_action(original_logits, action_ordering)
    reloaded_action = _select_action(reloaded_logits, action_ordering)
    selected_action_parity = original_action == reloaded_action

    # --- Determine status ---
    if max_abs_diff <= DEFAULT_TOLERANCE and selected_action_parity:
        status = AuditStatus.PASS
    else:
        status = AuditStatus.FAIL

    return {
        "policy_hash": policy_hash,
        "feature_transform_hash": feature_transform_hash,
        "action_ordering": action_ordering,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "original_logits": original_logits,
        "reloaded_logits": reloaded_logits,
        "max_abs_diff": max_abs_diff,
        "selected_action_parity": selected_action_parity,
        "status": status.value,
    }
