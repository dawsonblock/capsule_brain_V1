"""Provenance CLI for AutoLearn v0.3.1 (Section 25)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qualification.autolearn_v031.provenance import (
    build_provenance_manifest,
    compute_source_tree_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute provenance manifest.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    source_hash = compute_source_tree_sha256(root)
    print(f"source_tree_sha256: {source_hash}")

    manifest = build_provenance_manifest(root=root, package_version="2.15.2")
    (artifacts / "provenance_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )
    print(f"Written: {artifacts / 'provenance_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
