"""Provenance: SHA-256 chain over all v0.3.2 qualification artifacts (Section 13).

Computes source-tree hash, artifact digests, and a sequential proof chain so
that any tampering with an intermediate artifact is detectable. Each stage's
output is hashed into the next stage's input record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.source_hash import (
    compute_file_hash,
    compute_source_tree_hash,
    compute_string_hash,
    verify_provenance,
)

from .config import QualificationConfig
from .schemas import canonical_json, write_json


def _artifact_list(artifacts: Path) -> dict[str, str]:
    paths = [
        "benchmark_manifest.json",
        "split_manifest.json",
        "real_counterfactual_results.json",
        "executive_experiences.jsonl",
        "dataset_manifest.json",
        "dataset_train.json",
        "dataset_validation.json",
        "dataset_test.json",
        "dataset_ood.json",
        "dataset_safety.json",
        "candidate_policy.json",
        "candidate_training.json",
        "sham_policy.json",
        "sham_training.json",
        "evaluate_validation.json",
        "evaluate_test.json",
        "evaluate_ood.json",
        "evaluate_safety.json",
        "evaluate_oracle_test.json",
        "promotion_result.json",
        "post_promotion_result.json",
    ]
    d = {}
    for p in paths:
        full = artifacts / p
        d[p] = compute_file_hash(full) if full.exists() else "MISSING"
    return d


def build_provenance(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    here = Path(__file__).parent

    source_hash = compute_source_tree_hash(here)
    artifact_digests = _artifact_list(artifacts)

    # Build a sequential proof chain from artifact digests in dependency order.
    chain: list[dict] = []
    prev_hash = source_hash
    for name, h in artifact_digests.items():
        record = {"artifact": name, "hash": h, "previous": prev_hash}
        record_hash = compute_string_hash(canonical_json(record))
        chain.append({"record": record, "record_hash": record_hash})
        prev_hash = record_hash

    provenance = {
        "package_version": "2.15.3",
        "autolearn_qualification_version": "0.3.2",
        "source_hash": source_hash,
        "artifact_digests": artifact_digests,
        "chain": chain,
        "final_hash": chain[-1]["record_hash"] if chain else source_hash,
    }

    path = artifacts / "provenance.json"
    write_json(path, provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v0.3.2 provenance chain.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    provenance = build_provenance(config)
    print(f"provenance: source_hash={provenance['source_hash'][:16]}... final_hash={provenance['final_hash'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
