"""Generate provenance_manifest.json for AutoLearn v0.2.

Section 25: every scientific artifact must include hashes for:
- source_tree_sha256
- qualification_harness_sha256
- dataset_sha256
- split_manifest_sha256
- counterfactual_results_sha256
- policy_artifact_sha256
- evaluation_results_sha256

Uses repository-relative paths, not absolute local paths.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return _compute_sha256(path.read_bytes())


def _dir_sha256(path: Path) -> str:
    """Compute a deterministic hash of all files in a directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(path))
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _git_tree_sha(repo_root: Path) -> str:
    """Get the git tree SHA of the current commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    repo_root = out_dir.parent.parent  # qualification/autolearn_v02 -> repo root

    # Source tree hash (git commit SHA).
    source_tree_sha = _git_tree_sha(repo_root)

    # Qualification harness hash (all .py files in autolearn_v02/).
    harness_sha = _dir_sha256(out_dir)

    # Individual artifact hashes (repository-relative paths).
    artifacts = {
        "benchmark_manifest": "qualification/autolearn_v02/benchmark_manifest.json",
        "split_manifest": "qualification/autolearn_v02/split_manifest.json",
        "counterfactual_results": "qualification/autolearn_v02/counterfactuals.json",
        "dataset": "qualification/autolearn_v02/dataset.json",
        "dataset_manifest": "qualification/autolearn_v02/dataset_manifest.json",
        "evaluation_results": "qualification/autolearn_v02/evaluation.json",
        "shadow_eval": "qualification/autolearn_v02/shadow_eval.json",
        "promotion_result": "qualification/autolearn_v02/promotion_result.json",
        "policy_manifest": "qualification/autolearn_v02/policy_manifest.json",
    }
    artifact_hashes: dict[str, str | None] = {}
    for name, rel_path in artifacts.items():
        artifact_hashes[name] = _file_sha256(repo_root / rel_path)

    # Policy artifact hash (the actual policy JSON file, not the manifest).
    policy_id_file = out_dir / "candidate_policy_id.txt"
    policy_artifact_sha = None
    if policy_id_file.exists():
        policy_id = policy_id_file.read_text().strip()
        policy_path = out_dir / "policies" / "policies" / f"{policy_id}.json"
        policy_artifact_sha = _file_sha256(policy_path)

    manifest = {
        "source_tree_sha256": source_tree_sha,
        "qualification_harness_sha256": harness_sha,
        "dataset_sha256": artifact_hashes.get("dataset"),
        "split_manifest_sha256": artifact_hashes.get("split_manifest"),
        "counterfactual_results_sha256": artifact_hashes.get("counterfactual_results"),
        "policy_artifact_sha256": policy_artifact_sha,
        "evaluation_results_sha256": artifact_hashes.get("evaluation_results"),
        "benchmark_manifest_sha256": artifact_hashes.get("benchmark_manifest"),
        "dataset_manifest_sha256": artifact_hashes.get("dataset_manifest"),
        "shadow_eval_sha256": artifact_hashes.get("shadow_eval"),
        "promotion_result_sha256": artifact_hashes.get("promotion_result"),
        "policy_manifest_sha256": artifact_hashes.get("policy_manifest"),
        "paths": {
            k: v for k, v in artifacts.items()
        },
    }
    text = json.dumps(manifest, sort_keys=True, indent=2)
    (out_dir / "provenance_manifest.json").write_text(text)
    print(f"wrote {out_dir / 'provenance_manifest.json'}")
    print(f"source_tree_sha256: {source_tree_sha}")
    print(f"qualification_harness_sha256: {harness_sha[:16]}...")
    for k, v in artifact_hashes.items():
        if v:
            print(f"  {k}: {v[:16]}...")
        else:
            print(f"  {k}: MISSING")


if __name__ == "__main__":
    main()
