"""Sham policy trainer for v0.4.2 diagnostics.

Trains sham policies with various shuffle modes:
  global  — globally shuffle best-action labels
  family  — shuffle within task family
  margin  — shuffle within utility-margin bins
  feature — shuffle feature vectors, keep labels

Uses the same multinomial logistic regression as the candidate
to ensure architectural parity.
"""
from __future__ import annotations

import json
import numpy as np
from collections import Counter
from typing import Any

from .data import LoadedRun


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    actions: list[str],
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train multinomial logistic regression.

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    n_actions = len(actions)
    label_idx = {a: i for i, a in enumerate(actions)}
    y = np.array([label_idx.get(a, 0) for a in labels])

    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for epoch in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)
        # One-hot.
        one_hot = np.zeros((n, n_actions))
        one_hot[np.arange(n), y] = 1.0
        # Gradient.
        diff = probs - one_hot
        grad_W = features.T @ diff / n + l2 * W
        grad_b = diff.mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


def train_sham_policy(
    run: LoadedRun,
    shuffle_mode: str = "global",
    seed: int = 12345,
) -> dict[str, Any]:
    """Train a sham policy with the specified shuffle mode.

    Args:
        run: loaded run data.
        shuffle_mode: "global", "family", "margin", or "feature".
        seed: random seed for shuffling.

    Returns:
        Policy dict with weights, bias, actions, metadata.
    """
    examples = run.train_examples
    if not examples:
        return {"weights": [], "bias": [], "actions": run.all_actions, "error": "no training examples"}

    features = np.array([ex["features"] for ex in examples])
    labels = np.array([ex["best_action"] for ex in examples])
    actions = run.all_actions

    rng = np.random.default_rng(seed)

    if shuffle_mode == "global":
        # Globally shuffle labels.
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        train_labels = shuffled
    elif shuffle_mode == "family":
        # Shuffle within each family.
        train_labels = labels.copy()
        families = [ex.get("task_family", "unknown") for ex in examples]
        unique_families = set(families)
        for fam in unique_families:
            indices = [i for i, f in enumerate(families) if f == fam]
            if len(indices) > 1:
                fam_labels = train_labels[indices].copy()
                rng.shuffle(fam_labels)
                train_labels[indices] = fam_labels
    elif shuffle_mode == "margin":
        # Shuffle within margin bins.
        train_labels = labels.copy()
        margins = [ex.get("utility_margin", 0.0) for ex in examples]
        # Create bins.
        margins_arr = np.array(margins)
        bins = np.digitize(margins_arr, bins=[0.05, 0.25, 1.0])
        for bin_val in np.unique(bins):
            indices = np.where(bins == bin_val)[0]
            if len(indices) > 1:
                bin_labels = train_labels[indices].copy()
                rng.shuffle(bin_labels)
                train_labels[indices] = bin_labels
    elif shuffle_mode == "feature":
        # Shuffle feature vectors, keep labels.
        shuffled_idx = rng.permutation(len(features))
        features = features[shuffled_idx]
        train_labels = labels
    else:
        train_labels = labels

    W, b = train_logistic(features, train_labels, actions, seed=seed)

    return {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "actions": actions,
        "shuffle_mode": shuffle_mode,
        "seed": seed,
        "is_sham": True,
        "n_train": len(examples),
    }


def train_candidate_policy(
    run: LoadedRun,
    seed: int = 42,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
) -> dict[str, Any]:
    """Train a candidate policy (same architecture, real labels)."""
    examples = run.train_examples
    if not examples:
        return {"weights": [], "bias": [], "actions": run.all_actions, "error": "no training examples"}

    features = np.array([ex["features"] for ex in examples])
    labels = np.array([ex["best_action"] for ex in examples])
    actions = run.all_actions

    W, b = train_logistic(features, labels, actions, n_epochs=n_epochs, lr=lr, l2=l2, seed=seed)

    return {
        "weights": W.tolist(),
        "bias": b.tolist(),
        "actions": actions,
        "seed": seed,
        "is_sham": False,
        "n_train": len(examples),
        "n_epochs": n_epochs,
        "lr": lr,
        "l2": l2,
    }


def sham_ensemble(
    run: LoadedRun,
    n_seeds: int = 20,
    shuffle_mode: str = "global",
    base_seed: int = 10000,
) -> list[dict[str, Any]]:
    """Train multiple sham policies with different seeds."""
    return [
        train_sham_policy(run, shuffle_mode=shuffle_mode, seed=base_seed + i)
        for i in range(n_seeds)
    ]


def candidate_ensemble(
    run: LoadedRun,
    n_seeds: int = 20,
    base_seed: int = 20000,
) -> list[dict[str, Any]]:
    """Train multiple candidate policies with different seeds."""
    return [
        train_candidate_policy(run, seed=base_seed + i)
        for i in range(n_seeds)
    ]
