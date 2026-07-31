"""Preserve a completed scientific run as an immutable archived manifest.

Creates an immutable archive containing all provenance hashes and
metadata for the 7B full run, marked as a SCIENTIFIC NEGATIVE RESULT.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .data import load_run, _sha256_file, _sha256_json


def get_commit_sha(repo_root: str | Path) -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def archive_scientific_run(
    artifacts_dir: str | Path,
    archive_root: str | Path,
    run_id: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create an immutable archive manifest for a completed scientific run.

    Args:
        artifacts_dir: path to the run's artifacts.
        archive_root: root directory for archived runs.
        run_id: unique identifier for this run.
        repo_root: repository root for git SHA lookup.

    Returns:
        The archive manifest dict.
    """
    run = load_run(artifacts_dir)
    artifacts = Path(artifacts_dir)

    # Compute hashes for all key artifacts.
    artifact_hashes = {}
    for name in [
        "benchmark_manifest.json",
        "split_manifest.json",
        "counterfactual_outcomes.json",
        "dataset_manifest.json",
        "candidate_policy.json",
        "sham_policy.json",
        "gate_a_result.json",
        "promotion_result.json",
        "qualification_report.json",
        "runtime_completion_diagnostics.json",
        "candidate_training.json",
        "sham_training.json",
        "real_counterfactual_results.json",
    ]:
        p = artifacts / name
        if p.exists():
            artifact_hashes[name] = _sha256_file(p)

    # Evaluation result hashes.
    eval_hashes = {}
    for name in [
        "evaluate_test_baseline.json",
        "evaluate_test_candidate.json",
        "evaluate_test_sham.json",
        "evaluate_test_oracle.json",
        "evaluate_safety_candidate.json",
    ]:
        p = artifacts / name
        if p.exists():
            eval_hashes[name] = _sha256_file(p)

    # Extract model info from outcomes.
    model_id = ""
    model_revision = ""
    model_digest = ""
    provider_class = ""
    if run.outcomes:
        meta = run.outcomes[0].execution_metadata
        model_id = meta.get("model_id", "")
        model_revision = meta.get("model_revision", "")
        provider_class = meta.get("provider_class", "")

    # Gate results.
    gate_a = run.gate_a_result
    gate_b = {}
    gate_b_path = artifacts / "gate_b_result.json"
    if gate_b_path.exists():
        gate_b = json.loads(gate_b_path.read_text())

    commit_sha = get_commit_sha(repo_root) if repo_root else "unknown"

    manifest = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "package_version": run.qualification_report.get("package_version", "2.15.6"),
        "qualification_version": run.qualification_report.get("protocol_version", "0.4.2"),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_digest": model_digest,
        "provider_class": provider_class,
        "task_count": len(run.tasks),
        "counterfactual_count": len(run.outcomes),
        "random_seeds": {
            "task_seed": run.benchmark_manifest.get("task_seed", 42),
            "sham_seed": run.sham_policy.get("sham_seed"),
        },
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
        "split_manifest_sha256": run.split_manifest_sha256,
        "counterfactual_results_sha256": run.counterfactual_results_sha256,
        "dataset_sha256": run.dataset_sha256,
        "candidate_policy_sha256": artifact_hashes.get("candidate_policy.json", ""),
        "sham_policy_sha256": artifact_hashes.get("sham_policy.json", ""),
        "baseline_results_sha256": eval_hashes.get("evaluate_test_baseline.json", ""),
        "candidate_results_sha256": eval_hashes.get("evaluate_test_candidate.json", ""),
        "sham_results_sha256": eval_hashes.get("evaluate_test_sham.json", ""),
        "oracle_results_sha256": eval_hashes.get("evaluate_test_oracle.json", ""),
        "gate_a_result": {
            "status": gate_a.get("status"),
            "candidate_mean_utility": gate_a.get("candidate_mean_utility"),
            "baseline_mean_utility": gate_a.get("baseline_mean_utility"),
            "sham_mean_utility": gate_a.get("sham_mean_utility"),
            "oracle_mean_utility": gate_a.get("oracle_mean_utility"),
            "delta_candidate": gate_a.get("delta_candidate"),
            "delta_candidate_sham": gate_a.get("delta_candidate_sham"),
        },
        "gate_b_result": {
            "status": gate_b.get("status", "NOT_RUN"),
        } if gate_b else {"status": "NOT_RUN"},
        "final_report_sha256": artifact_hashes.get("qualification_report.json", ""),
        "artifact_hashes": artifact_hashes,
        "eval_hashes": eval_hashes,
        "mark": "IMMUTABLE",
        "result_classification": "SCIENTIFIC NEGATIVE RESULT",
        "promotion_status": "NOT PROMOTED",
        "archive_created_at": _timestamp(),
    }

    # Write to archive location.
    archive_dir = Path(archive_root) / "scientific" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Write an IMMUTABLE marker.
    (archive_dir / "IMMUTABLE").write_text(
        f"This run is archived as IMMUTABLE.\n"
        f"Run ID: {run_id}\n"
        f"Classification: SCIENTIFIC NEGATIVE RESULT\n"
        f"Promotion: NOT PROMOTED\n"
        f"Do not modify or overwrite.\n"
    )

    return manifest


def _timestamp() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
