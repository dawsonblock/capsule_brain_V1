"""Modal evidence recovery command for v0.4.6.

Searches a source directory for required modal evidence files.
If required files are missing, it stops, does NOT synthesize anything,
and writes a MISSING_MODAL_EVIDENCE.json report to the output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus


# Required modal evidence files to search for.
# Each entry maps a canonical name to a list of acceptable filename patterns.
REQUIRED_MODAL_FILES: list[tuple[str, tuple[str, ...]]] = [
    ("benchmark_manifest", ("benchmark_manifest.json",)),
    ("split_manifest", ("split_manifest.json",)),
    ("provider_manifest", ("provider_manifest.json",)),
    ("generation_config", ("generation_config.json", "generation_config.yaml")),
    ("utility_config", ("utility_config.json", "utility_config.yaml")),
    ("counterfactual_outcomes", ("counterfactual_outcomes.jsonl", "counterfactual_outcomes.json")),
    ("experience_rows", ("executive_experiences.jsonl", "experiences.jsonl", "experience_rows.jsonl")),
    ("candidate_policy", ("candidate_policy.json",)),
    ("sham_policy", ("sham_policy.json",)),
    ("baseline_decisions", ("baseline_results.json", "baseline_decisions.json")),
    ("candidate_decisions", ("candidate_results.json", "candidate_decisions.json")),
    ("sham_decisions", ("sham_results.json", "sham_decisions.json")),
    ("oracle_decisions_or_action_matrix", (
        "oracle_results.json",
        "oracle_decisions.json",
        "action_matrix.json",
        "complete_action_matrix.json",
    )),
    ("candidate_decision_trace", (
        "candidate_decision_trace.json",
        "candidate_trace.jsonl",
        "candidate_decision_trace.jsonl",
    )),
    ("safety_rows", ("safety_results.jsonl", "safety_results.json")),
    ("runtime_completion_diagnostics", (
        "runtime_completion_diagnostics.json",
        "runtime_diagnostics.json",
    )),
    ("source_provenance", ("source_provenance.json",)),
    ("original_config", (
        "original_config.json",
        "config.json",
        "run_config.json",
    )),
    ("dependency_lock_identity", (
        "dependency_lock.json",
        "requirements_lock.txt",
        "poetry.lock",
        "uv.lock",
    )),
]


def _find_file(source_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """Search for a file matching any of the given patterns in source_dir.

    Also searches one level of subdirectories.
    """
    for pattern in patterns:
        # Direct match
        candidate = source_dir / pattern
        if candidate.exists() and candidate.is_file():
            return candidate
        # One level deep
        for sub in source_dir.iterdir():
            if not sub.is_dir():
                continue
            candidate = sub / pattern
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _write_missing_report(
    output_dir: Path,
    missing_files: list[str],
    recovered_files: dict[str, str],
    expected_run_id: str,
    reason: str,
) -> None:
    """Write a MISSING_MODAL_EVIDENCE.json report."""
    report = {
        "status": "MISSING",
        "expected_run_id": expected_run_id,
        "missing_files": missing_files,
        "recovered_files": recovered_files,
        "n_missing": len(missing_files),
        "n_recovered": len(recovered_files),
        "reason": reason,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "MISSING_MODAL_EVIDENCE.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def recover_modal_evidence(
    source_dir: str,
    output_dir: str,
    expected_run_id: str = "scientific_full_7b_001",
) -> dict[str, Any]:
    """Recover modal evidence from a source directory.

    Searches for all required modal evidence files.  If any required files
    are missing, stops immediately, does NOT synthesize anything, and writes
    a MISSING_MODAL_EVIDENCE.json report to output_dir.

    Args:
        source_dir: Directory to search for evidence files.
        output_dir: Directory to write recovered evidence and reports.
        expected_run_id: Expected run ID for the evidence.

    Returns:
        dict with status (RECOVERED or MISSING), recovered_files,
        missing_files, output_dir, reason
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    recovered_files: dict[str, str] = {}
    missing_files: list[str] = []

    # Check source directory exists
    if not source_path.exists() or not source_path.is_dir():
        missing_names = [name for name, _ in REQUIRED_MODAL_FILES]
        reason = f"Source directory does not exist or is not a directory: {source_dir}"
        _write_missing_report(output_path, missing_names, {}, expected_run_id, reason)
        return {
            "status": "MISSING",
            "recovered_files": {},
            "missing_files": missing_names,
            "output_dir": str(output_path),
            "reason": reason,
        }

    # Search for each required file
    for canonical_name, patterns in REQUIRED_MODAL_FILES:
        found = _find_file(source_path, patterns)
        if found is not None:
            recovered_files[canonical_name] = str(found)
        else:
            missing_files.append(canonical_name)

    # If any required files are missing, stop and write report
    if missing_files:
        reason = (
            f"Missing {len(missing_files)} required modal evidence file(s): "
            f"{', '.join(missing_files)}. "
            f"Recovery stopped — no evidence synthesized."
        )
        _write_missing_report(output_path, missing_files, recovered_files, expected_run_id, reason)
        return {
            "status": "MISSING",
            "recovered_files": recovered_files,
            "missing_files": missing_files,
            "output_dir": str(output_path),
            "reason": reason,
        }

    # All required files found — copy them to output directory
    output_path.mkdir(parents=True, exist_ok=True)
    import shutil

    copied_files: dict[str, str] = {}
    for canonical_name, src_str in recovered_files.items():
        src = Path(src_str)
        dst = output_path / src.name
        try:
            shutil.copy2(src, dst)
            copied_files[canonical_name] = str(dst)
        except OSError:
            # If copy fails, treat as missing
            missing_files.append(canonical_name)

    if missing_files:
        reason = (
            f"Failed to copy {len(missing_files)} file(s): "
            f"{', '.join(missing_files)}"
        )
        _write_missing_report(output_path, missing_files, copied_files, expected_run_id, reason)
        return {
            "status": "MISSING",
            "recovered_files": copied_files,
            "missing_files": missing_files,
            "output_dir": str(output_path),
            "reason": reason,
        }

    reason = (
        f"Recovered {len(copied_files)} modal evidence file(s) "
        f"for run_id={expected_run_id}"
    )
    return {
        "status": "RECOVERED",
        "recovered_files": copied_files,
        "missing_files": [],
        "output_dir": str(output_path),
        "reason": reason,
    }
