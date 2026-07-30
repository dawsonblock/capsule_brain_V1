"""Gate B: Passive latent verification via CLUE (Section 12).

CLUE (Confidence Latent Utility Estimator) reads hidden-state activation
deltas from a real model's forward pass WITHOUT altering generation
(Section 12.1 passivity). It trains success/failure prototypes on the
experience split and scores held-out test trajectories.

This module implements:
  * the CLUE prototype model (Section 12.2);
  * distance-based scoring (euclidean / cosine / mahalanobis);
  * non-latent confidence baselines (max-softmax, token-frequency, prompt
    length, action-identity) (Section 12.10);
  * AUROC / AUPRC / Brier / ECE metrics (Section 12.11);
  * pass criteria (Section 12.12).

Gate B is BLOCKED when:
  * the provider is the infrastructure-only deterministic provider (no
    hidden states);
  * no real model backend is available;
  * activation artifacts are missing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import read_json, sha256_json, write_json
from .statistics import (
    auroc,
    auprc,
    balanced_accuracy,
    bootstrap_auroc_ci,
    brier_score,
    expected_calibration_error,
)


@dataclass(slots=True)
class CLUEPrototype:
    """A success/failure prototype centroid over hidden states."""
    layer_id: int
    success_centroid: np.ndarray
    failure_centroid: np.ndarray
    success_inv_cov: np.ndarray | None = None
    failure_inv_cov: np.ndarray | None = None
    n_success: int = 0
    n_failure: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "n_success": self.n_success,
            "n_failure": self.n_failure,
            "success_centroid_norm": float(np.linalg.norm(self.success_centroid)),
            "failure_centroid_norm": float(np.linalg.norm(self.failure_centroid)),
        }


def _euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _mahalanobis_dist(a: np.ndarray, centroid: np.ndarray, inv_cov: np.ndarray | None) -> float:
    if inv_cov is None:
        return _euclidean_dist(a, centroid)
    diff = a - centroid
    return float(np.sqrt(max(0.0, diff @ inv_cov @ diff)))


def _distance(a: np.ndarray, b: np.ndarray, metric: str, inv_cov: np.ndarray | None = None) -> float:
    if metric == "euclidean":
        return _euclidean_dist(a, b)
    if metric == "cosine":
        return _cosine_dist(a, b)
    if metric == "mahalanobis":
        return _mahalanobis_dist(a, b, inv_cov)
    raise ValueError(f"unknown distance metric: {metric}")


def train_prototypes(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    layer_ids: list[int],
    *,
    distance_metric: str = "euclidean",
) -> dict[int, CLUEPrototype]:
    """Train success/failure prototypes per layer on the experience split."""
    prototypes: dict[int, CLUEPrototype] = {}
    for layer_id in layer_ids:
        acts = activations.get(layer_id)
        if acts is None or len(acts) == 0:
            continue
        success_mask = labels == 1
        failure_mask = labels == 0
        if success_mask.sum() == 0 or failure_mask.sum() == 0:
            continue
        success_centroid = acts[success_mask].mean(axis=0)
        failure_centroid = acts[failure_mask].mean(axis=0)
        success_inv_cov = None
        failure_inv_cov = None
        if distance_metric == "mahalanobis":
            try:
                success_inv_cov = np.linalg.pinv(np.cov(acts[success_mask].T) + 1e-6 * np.eye(acts.shape[1]))
                failure_inv_cov = np.linalg.pinv(np.cov(acts[failure_mask].T) + 1e-6 * np.eye(acts.shape[1]))
            except Exception:
                success_inv_cov = None
                failure_inv_cov = None
        prototypes[layer_id] = CLUEPrototype(
            layer_id=layer_id,
            success_centroid=success_centroid,
            failure_centroid=failure_centroid,
            success_inv_cov=success_inv_cov,
            failure_inv_cov=failure_inv_cov,
            n_success=int(success_mask.sum()),
            n_failure=int(failure_mask.sum()),
        )
    return prototypes


def score_trajectory(
    activation: np.ndarray,
    prototype: CLUEPrototype,
    *,
    distance_metric: str = "euclidean",
) -> float:
    """Score a trajectory: P(success) = sigmoid(d_failure - d_success)."""
    d_success = _distance(activation, prototype.success_centroid, distance_metric, prototype.success_inv_cov)
    d_failure = _distance(activation, prototype.failure_centroid, distance_metric, prototype.failure_inv_cov)
    return 1.0 / (1.0 + np.exp(-(d_failure - d_success)))


def score_all(
    activations: dict[int, np.ndarray],
    prototypes: dict[int, CLUEPrototype],
    *,
    distance_metric: str = "euclidean",
) -> dict[int, np.ndarray]:
    """Score all trajectories for each layer."""
    scores: dict[int, np.ndarray] = {}
    for layer_id, proto in prototypes.items():
        acts = activations.get(layer_id)
        if acts is None:
            continue
        scores[layer_id] = np.array([
            score_trajectory(a, proto, distance_metric=distance_metric) for a in acts
        ])
    return scores


# ---------------------------------------------------------------------------
# Non-latent confidence baselines (Section 12.10)
# ---------------------------------------------------------------------------


def max_softmax_baseline(logprobs: list[float]) -> np.ndarray:
    """Max-softmax confidence from token log-probs."""
    if not logprobs:
        return np.array([])
    arr = np.array(logprobs)
    # softmax over the sequence of log-probs (per-token confidence proxy)
    e = np.exp(arr - arr.max())
    return e / e.sum() if e.sum() > 0 else np.full_like(arr, 0.5)


def token_frequency_baseline(token_ids: list[int], train_token_freq: dict[int, float]) -> float:
    """Average training-set token frequency as a confidence proxy."""
    if not token_ids:
        return 0.5
    freqs = [train_token_freq.get(int(t), 0.0) for t in token_ids]
    return float(np.mean(freqs)) if freqs else 0.5


def prompt_length_baseline(prompt_lengths: list[int]) -> np.ndarray:
    """Longer prompts => higher confidence proxy (weak baseline)."""
    if not prompt_lengths:
        return np.array([])
    arr = np.array(prompt_lengths, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)


def action_identity_baseline(action_ids: list[str], success_action_freq: dict[str, float]) -> np.ndarray:
    """Action-identity confidence: P(success | action) from training."""
    return np.array([success_action_freq.get(a, 0.5) for a in action_ids])


# ---------------------------------------------------------------------------
# Gate B evaluation
# ---------------------------------------------------------------------------


def evaluate_gate_b(config: QualificationConfig) -> dict[str, Any]:
    """Run the full Gate B CLUE evaluation."""
    artifacts = Path(config.artifacts_dir)

    # Block if infrastructure-only provider (no hidden states).
    if config.is_infrastructure_provider:
        return _blocked_result(
            "Gate B requires hidden-state activations from a real model "
            "backend. The infrastructure-only deterministic provider has "
            "no hidden states."
        )

    activations_path = artifacts / "activations_manifest.json"
    if not activations_path.exists():
        return _blocked_result("activations_manifest.json missing; run collect_activations first")

    manifest = read_json(activations_path)
    if not manifest.get("trajectories"):
        return _blocked_result("no activation trajectories collected")

    # Load the activation arrays.
    layer_ids = manifest.get("layer_ids", [])
    if not layer_ids:
        return _blocked_result("no layer_ids in activation manifest")

    train_acts = _load_activation_arrays(manifest, "experience", layer_ids, artifacts)
    test_acts = _load_activation_arrays(manifest, "test", layer_ids, artifacts)
    train_labels = np.array(manifest.get("train_labels", []), dtype=int)
    test_labels = np.array(manifest.get("test_labels", []), dtype=int)

    if train_acts is None or test_acts is None:
        return _blocked_result("could not load activation arrays")
    if len(train_labels) == 0 or len(test_labels) == 0:
        return _blocked_result("missing train or test labels")

    # Train prototypes.
    prototypes = train_prototypes(
        train_acts, train_labels, layer_ids,
        distance_metric=config.gate_b_distance_metric,
    )
    if not prototypes:
        return _blocked_result("could not train prototypes (insufficient class diversity)")

    # Score test trajectories.
    test_scores = score_all(
        test_acts, prototypes,
        distance_metric=config.gate_b_distance_metric,
    )

    # Pick the best layer by AUROC on test (Section 12.11).
    layer_results: dict[int, dict[str, Any]] = {}
    best_layer = None
    best_auroc = -1.0
    for layer_id, scores in test_scores.items():
        auroc_val = auroc(scores.tolist(), test_labels.tolist())
        auprc_val = auprc(scores.tolist(), test_labels.tolist())
        brier = brier_score(scores.tolist(), test_labels.tolist())
        ece = expected_calibration_error(scores.tolist(), test_labels.tolist())
        bal_acc = balanced_accuracy(scores.tolist(), test_labels.tolist())
        point, ci_low, ci_high = bootstrap_auroc_ci(scores.tolist(), test_labels.tolist())
        layer_results[layer_id] = {
            "auroc": auroc_val,
            "auprc": auprc_val,
            "brier": brier,
            "ece": ece,
            "balanced_accuracy": bal_acc,
            "auroc_ci_low": ci_low,
            "auroc_ci_high": ci_high,
        }
        if auroc_val > best_auroc:
            best_auroc = auroc_val
            best_layer = layer_id

    # Non-latent baselines.
    train_token_freq = manifest.get("train_token_freq", {})
    success_action_freq = manifest.get("success_action_freq", {})
    test_logprobs = manifest.get("test_logprobs", [])
    test_token_ids = manifest.get("test_token_ids", [])
    test_prompt_lengths = manifest.get("test_prompt_lengths", [])
    test_action_ids = manifest.get("test_action_ids", [])

    nonlatent_baselines: dict[str, float] = {}
    if test_logprobs:
        ms_scores = max_softmax_baseline(test_logprobs)
        if len(ms_scores) == len(test_labels):
            nonlatent_baselines["max_softmax"] = auroc(ms_scores.tolist(), test_labels.tolist())
    if test_token_ids and train_token_freq:
        tf_scores = np.array([
            token_frequency_baseline(tids, train_token_freq) for tids in test_token_ids
        ])
        if len(tf_scores) == len(test_labels):
            nonlatent_baselines["token_frequency"] = auroc(tf_scores.tolist(), test_labels.tolist())
    if test_prompt_lengths:
        pl_scores = prompt_length_baseline(test_prompt_lengths)
        if len(pl_scores) == len(test_labels):
            nonlatent_baselines["prompt_length"] = auroc(pl_scores.tolist(), test_labels.tolist())
    if test_action_ids:
        ai_scores = action_identity_baseline(test_action_ids, success_action_freq)
        if len(ai_scores) == len(test_labels):
            nonlatent_baselines["action_identity"] = auroc(ai_scores.tolist(), test_labels.tolist())

    best_nonlatent_auroc = max(nonlatent_baselines.values()) if nonlatent_baselines else 0.5
    chance_auroc = 0.5
    margin_over_chance = best_auroc - chance_auroc
    margin_over_best_nonlatent = best_auroc - best_nonlatent_auroc

    # Pass criteria (Section 12.12).
    from .schemas import GateCategory, GateResult, GateStatus
    gates: list[GateResult] = []
    gates.append(GateResult(
        "gate_b_auroc_above_chance", GateCategory.STATISTICAL_SUPERIORITY,
        GateStatus.PASS if margin_over_chance > config.gate_b_minimum_auroc_margin_over_chance else GateStatus.FAIL,
        {"auroc": best_auroc, "chance": chance_auroc, "margin": margin_over_chance,
         "required_margin": config.gate_b_minimum_auroc_margin_over_chance},
    ))
    if config.gate_b_require_superiority_over_best_nonlatent:
        gates.append(GateResult(
            "gate_b_superior_to_best_nonlatent", GateCategory.STATISTICAL_SUPERIORITY,
            GateStatus.PASS if margin_over_best_nonlatent > 0 else GateStatus.FAIL,
            {"clue_auroc": best_auroc, "best_nonlatent_auroc": best_nonlatent_auroc,
             "margin": margin_over_best_nonlatent},
        ))
    gates.append(GateResult(
        "gate_b_best_layer_identified", GateCategory.EVIDENCE_COMPLETENESS,
        GateStatus.PASS if best_layer is not None else GateStatus.FAIL,
        {"best_layer": best_layer},
    ))
    gates.append(GateResult(
        "gate_b_calibration", GateCategory.EVIDENCE_COMPLETENESS,
        GateStatus.PASS if (layer_results.get(best_layer, {}).get("ece", 1.0) < 0.3) else GateStatus.FAIL,
        {"ece": layer_results.get(best_layer, {}).get("ece", 1.0)},
    ))

    n_passed = sum(1 for g in gates if g.status is GateStatus.PASS)
    n_failed = sum(1 for g in gates if g.status is GateStatus.FAIL)
    all_passed = n_failed == 0

    result = {
        "schema_version": "gate-b-result/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "PASS" if all_passed else "FAIL",
        "provider": config.provider,
        "model": config.model,
        "is_infrastructure_provider": config.is_infrastructure_only,
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "margin_over_chance": margin_over_chance,
        "best_nonlatent_auroc": best_nonlatent_auroc,
        "margin_over_best_nonlatent": margin_over_best_nonlatent,
        "layer_results": layer_results,
        "nonlatent_baselines": nonlatent_baselines,
        "prototypes": {lid: p.to_dict() for lid, p in prototypes.items()},
        "gates": [g.to_dict() for g in gates],
        "primary_gate_count": len(gates),
        "primary_gates_passed": n_passed,
        "primary_gates_failed": n_failed,
        "summary_decision": "PASS" if all_passed else "FAIL",
    }
    write_json(artifacts / "gate_b_result.json", result)
    return result


def _load_activation_arrays(
    manifest: dict, split: str, layer_ids: list[int], artifacts: Path
) -> dict[int, np.ndarray] | None:
    """Load activation arrays from .npy files referenced in the manifest."""
    arrays: dict[int, np.ndarray] = {}
    trajs = [t for t in manifest.get("trajectories", []) if t.get("split_name") == split]
    if not trajs:
        return None
    for layer_id in layer_ids:
        rows: list[np.ndarray] = []
        for t in trajs:
            file_field = f"activation_file_layer_{layer_id}"
            fname = t.get(file_field) or t.get("activation_file", "")
            if not fname:
                continue
            fpath = artifacts / fname
            if not fpath.exists():
                continue
            arr = np.load(fpath)
            rows.append(arr)
        if rows:
            arrays[layer_id] = np.stack(rows)
    return arrays if arrays else None


def _blocked_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "gate-b-result/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "BLOCKED",
        "reason": reason,
        "is_infrastructure_provider": True,
        "gates": [],
        "primary_gate_count": 0,
        "primary_gates_passed": 0,
        "primary_gates_failed": 0,
        "summary_decision": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gate B (passive CLUE).")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    parser.add_argument("--model", default="qual-grounded-v04")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir,
    )
    result = evaluate_gate_b(config)
    print(f"Gate B: {result['status']}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
