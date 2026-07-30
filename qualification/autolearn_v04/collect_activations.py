"""Passive hidden-state activation collection for Gate B (Section 12.7).

Reads hidden states from a real model's forward pass WITHOUT altering
generation. The model generates normally; hidden states for selected
layers are extracted from the SAME forward pass (Section 12.1 passivity).

Activations are saved as .npy files; a manifest records trajectory
metadata, layer ids, verifier labels, and the model execution identity.

This module is BLOCKED for the infrastructure-only deterministic provider
(no hidden states).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import AUTOLEARN_QUALIFICATION_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import PROVIDER_INFRASTRUCTURE, PROVIDER_LOCAL_TRANSFORMERS, QualificationConfig
from .schemas import (
    ActivationTrace,
    OutcomeAvailability,
    VerificationOutcome,
    read_json,
    sha256_text,
    write_json,
)


def _select_layers(n_layers: int, strategy: str) -> list[int]:
    """Select layer ids per Section 12.6 boundary method."""
    if n_layers == 0:
        return []
    if strategy == "late_quartile":
        start = max(0, n_layers - max(1, n_layers // 4))
        return list(range(start, n_layers))
    if strategy == "all":
        return list(range(n_layers))
    if strategy == "middle":
        mid = n_layers // 2
        return [mid] if mid < n_layers else [n_layers - 1]
    return [n_layers - 1]


async def collect_activations(config: QualificationConfig) -> dict[str, Any]:
    """Collect passive hidden-state activations from a real model."""
    artifacts = Path(config.artifacts_dir)

    if config.is_infrastructure_provider:
        return _blocked_manifest(
            "Activation collection requires a real model backend with "
            "hidden states. The infrastructure-only deterministic provider "
            "has no hidden states."
        )

    # Load the benchmark and counterfactual outcomes.
    bm = read_json(artifacts / "benchmark_manifest.json")
    tasks_by_id = {t["task_id"]: t for t in bm["tasks"]}
    outcomes = read_json(artifacts / "counterfactual_outcomes.json")

    # Filter to EXECUTED outcomes with verification labels.
    executed = [
        o for o in outcomes
        if o.get("availability") == "executed"
        and o.get("verification") in ("success", "failure")
    ]

    # Build the real model provider.
    from .local_transformers_provider import LocalTransformersProvider
    provider = LocalTransformersProvider(
        model_id=config.model,
        device=config.device,
        dtype=config.dtype,
        collect_hidden_states=True,
        hidden_layer_ids=[],  # will set after loading
    )

    n_layers = provider._model.config.num_hidden_layers if hasattr(provider._model, "config") else 0
    layer_ids = _select_layers(n_layers, config.gate_b_layers)
    provider.hidden_layer_ids = layer_ids

    model_identity = provider.model_execution_identity

    trajectories: list[dict[str, Any]] = []
    train_labels: list[int] = []
    test_labels: list[int] = []
    train_token_freq: dict[int, float] = {}
    success_action_freq: dict[str, float] = {}
    train_token_counts: dict[int, int] = {}
    train_total_tokens = 0
    train_success_by_action: dict[str, int] = {}
    train_action_counts: dict[str, int] = {}

    # Collect activations for each executed outcome.
    from capsule_brain.llm.models import LLMRequest
    for o in executed:
        task = tasks_by_id.get(o["task_id"])
        if task is None:
            continue
        split = task.get("split", "experience")
        if split not in ("experience", "test"):
            continue
        prompt = task["prompt"]
        request = LLMRequest(
            prompt=prompt,
            model=config.model_id,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            temperature=0.0,
            max_tokens=512,
        )
        try:
            result = await provider.generate(request, {})
        except Exception:
            continue

        label = 1 if o.get("verification") == "success" else 0
        token_ids = provider.last_token_ids
        logprobs = provider.last_logprobs
        hidden = provider.last_hidden_states

        if not hidden:
            continue

        traj_id = f"act_{o['task_id']}_{o['action_id']}"
        # Save activation arrays per layer.
        for layer_id, arr in hidden.items():
            fname = f"activations/{traj_id}_layer_{layer_id}.npy"
            fpath = artifacts / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(fpath, arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr))

        trajectories.append({
            "trajectory_id": traj_id,
            "task_id": o["task_id"],
            "action_id": o["action_id"],
            "split_name": split,
            "layer_ids": list(hidden.keys()),
            "activation_file": f"activations/{traj_id}_layer_{list(hidden.keys())[0]}.npy",
            "verifier_label": label,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "prompt_length": len(prompt),
        })

        if split == "experience":
            train_labels.append(label)
            for tid in token_ids:
                train_token_counts[int(tid)] = train_token_counts.get(int(tid), 0) + 1
                train_total_tokens += 1
            action = o["action_id"]
            train_action_counts[action] = train_action_counts.get(action, 0) + 1
            if label == 1:
                train_success_by_action[action] = train_success_by_action.get(action, 0) + 1
        elif split == "test":
            test_labels.append(label)

    # Compute training-set statistics for non-latent baselines.
    if train_total_tokens > 0:
        train_token_freq = {k: v / train_total_tokens for k, v in train_token_counts.items()}
    success_action_freq = {
        a: (train_success_by_action.get(a, 0) / train_action_counts[a])
        for a in train_action_counts if train_action_counts[a] > 0
    }

    # Add per-layer activation file references to trajectories.
    for t in trajectories:
        for lid in layer_ids:
            t[f"activation_file_layer_{lid}"] = (
                f"activations/{t['trajectory_id']}_layer_{lid}.npy"
            )

    manifest = {
        "schema_version": "activations-manifest/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "model_identity": model_identity.to_dict(),
        "model_identity_digest": model_identity.digest,
        "layer_ids": layer_ids,
        "n_trajectories": len(trajectories),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_token_freq": train_token_freq,
        "success_action_freq": success_action_freq,
        "test_logprobs": [t.get("logprobs", []) for t in trajectories if t["split_name"] == "test"],
        "test_token_ids": [t.get("token_ids", []) for t in trajectories if t["split_name"] == "test"],
        "test_prompt_lengths": [t.get("prompt_length", 0) for t in trajectories if t["split_name"] == "test"],
        "test_action_ids": [t.get("action_id", "") for t in trajectories if t["split_name"] == "test"],
        "trajectories": trajectories,
        "passivity_guaranteed": True,
        "boundary_method": config.gate_b_layers,
    }
    write_json(artifacts / "activations_manifest.json", manifest)
    await provider.close()
    return manifest


def _blocked_manifest(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "activations-manifest/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "BLOCKED",
        "reason": reason,
        "trajectories": [],
        "layer_ids": [],
        "n_trajectories": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect passive hidden-state activations for Gate B.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="local_transformers")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir,
    )
    manifest = asyncio.run(collect_activations(config))
    print(f"activations: {manifest.get('n_trajectories', 0)} trajectories")
    if manifest.get("reason"):
        print(f"  BLOCKED: {manifest['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
