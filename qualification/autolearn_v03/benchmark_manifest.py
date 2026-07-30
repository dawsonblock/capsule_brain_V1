"""Generate benchmark_manifest.json for AutoLearn v0.3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tasks import build_all_tasks, build_split_assignments, get_archetype_summary


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)
    summary = get_archetype_summary(tasks)

    from collections import Counter
    split_counts = Counter(assignments.values())

    manifest = {
        "benchmark_version": "autolearn_v0.3",
        "task_count": len(tasks),
        "archetype_count": len(summary),
        "split_strategy": "archetype_based",
        "split_counts": dict(split_counts),
        "feature_schema_version": "exec_features_v2",
        "utility_config_version": "exec_utility_v3",
        "schema_version": "exec_experience_v2",
        "archetypes": summary,
        "ood_archetypes": [
            a for a, counts in summary.items()
            if counts.get("ood", 0) > 0
        ],
        "learned_actions": ["ANSWER_DIRECT", "RETRIEVE_MEMORY", "CALL_TOOL", "START_WORKFLOW"],
        "runtime_controlled_actions": ["REFLECT", "ASK_OPERATOR"],
    }
    text = json.dumps(manifest, sort_keys=True, indent=2)
    (out_dir / "benchmark_manifest.json").write_text(text)
    print(f"wrote {out_dir / 'benchmark_manifest.json'}")
    print(f"benchmark SHA-256: {_compute_sha256(text)}")


if __name__ == "__main__":
    main()
