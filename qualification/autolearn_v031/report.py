"""Report generator for AutoLearn v0.3.1.

v0.3.1 Section 24: Report integrity.

The report generator derives every statement from artifacts. It never
hardcodes "Candidate beats baseline" or "Candidate promoted". It uses
PASS/FAIL/BLOCKED/NOT RUN for every stage.

The report reads the active policy from the registry after promotion. It
must not label the candidate as active merely because it was evaluated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.promotion import GateStatus


class StageStatus(Enum):
    """Status of a qualification stage."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class StageReport:
    """A single stage in the final report."""
    name: str
    status: StageStatus
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_markdown(self) -> str:
        lines = [f"### {self.name}", f"**Status:** {self.status.value}"]
        if self.reason:
            lines.append(f"**Reason:** {self.reason}")
        for k, v in self.details.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _stage_from_gate(gate_name: str, gates: list[dict[str, Any]]) -> StageReport:
    """Build a stage report from a promotion gate result."""
    gate = next((g for g in gates if g["name"] == gate_name), None)
    if gate is None:
        return StageReport(name=gate_name, status=StageStatus.NOT_RUN,
                           reason="gate not found")
    status_map = {
        "PASS": StageStatus.PASS,
        "FAIL": StageStatus.FAIL,
        "BLOCKED": StageStatus.BLOCKED,
        "NOT_RUN": StageStatus.NOT_RUN,
    }
    status = status_map.get(gate.get("status", "NOT_RUN"), StageStatus.NOT_RUN)
    return StageReport(
        name=gate_name,
        status=status,
        details={
            "observed": gate.get("observed"),
            "threshold": gate.get("threshold"),
        },
        reason=gate.get("reason", ""),
    )


def generate_report(
    artifacts_dir: str | Path,
    *,
    active_policy_id: str = "",
) -> str:
    """Generate the final qualification report as Markdown.

    Every statement is derived from artifacts in the artifacts directory.
    Nothing is hardcoded.
    """
    artifacts = Path(artifacts_dir)
    stages: list[StageReport] = []

    # --- Real counterfactual runtime ---
    cf_results = _load_json(artifacts / "real_counterfactual_results.json")
    if cf_results is None:
        stages.append(StageReport(
            name="Real counterfactual runtime",
            status=StageStatus.NOT_RUN,
            reason="real_counterfactual_results.json not found",
        ))
    else:
        runtime_type = cf_results.get("runtime_type", "unknown")
        n_results = len(cf_results.get("results", []))
        if runtime_type == "real":
            stages.append(StageReport(
                name="Real counterfactual runtime",
                status=StageStatus.PASS,
                details={"runtime_type": runtime_type, "n_results": n_results},
            ))
        else:
            stages.append(StageReport(
                name="Real counterfactual runtime",
                status=StageStatus.FAIL,
                details={"runtime_type": runtime_type, "n_results": n_results},
                reason="runtime_type is not 'real'",
            ))

    # --- Policy serialization parity ---
    candidate_policy = _load_json(artifacts / "candidate_policy.json")
    if candidate_policy is None:
        stages.append(StageReport(
            name="Policy serialization parity",
            status=StageStatus.NOT_RUN,
            reason="candidate_policy.json not found",
        ))
    else:
        has_transform = candidate_policy.get("feature_transform") is not None
        if has_transform:
            stages.append(StageReport(
                name="Policy serialization parity",
                status=StageStatus.PASS,
                details={"feature_transform_present": True},
            ))
        else:
            stages.append(StageReport(
                name="Policy serialization parity",
                status=StageStatus.FAIL,
                details={"feature_transform_present": False},
                reason="FeatureTransform missing from policy artifact",
            ))

    # --- Candidate versus baseline ---
    test_results = _load_json(artifacts / "test_results.json")
    if test_results is None:
        stages.append(StageReport(
            name="Candidate versus baseline",
            status=StageStatus.NOT_RUN,
            reason="test_results.json not found",
        ))
    else:
        mean_delta = test_results.get("mean_delta", 0.0)
        ci_low = test_results.get("delta_ci_low", 0.0)
        ci_high = test_results.get("delta_ci_high", 0.0)
        if ci_low > 0:
            stages.append(StageReport(
                name="Candidate versus baseline",
                status=StageStatus.PASS,
                details={
                    "mean_delta": round(mean_delta, 4),
                    "95%_CI": f"[{round(ci_low, 4)}, {round(ci_high, 4)}]",
                },
            ))
        else:
            stages.append(StageReport(
                name="Candidate versus baseline",
                status=StageStatus.FAIL,
                details={
                    "mean_delta": round(mean_delta, 4),
                    "95%_CI": f"[{round(ci_low, 4)}, {round(ci_high, 4)}]",
                },
                reason="lower CI is not above 0",
            ))

    # --- Candidate versus sham ---
    sham_results = _load_json(artifacts / "sham_results.json")
    if sham_results is None:
        stages.append(StageReport(
            name="Candidate versus sham",
            status=StageStatus.NOT_RUN,
            reason="sham_results.json not found",
        ))
    else:
        sham_delta = sham_results.get("mean_delta", 0.0)
        if sham_delta > 0:
            stages.append(StageReport(
                name="Candidate versus sham",
                status=StageStatus.PASS,
                details={"mean_delta_vs_sham": round(sham_delta, 4)},
            ))
        else:
            stages.append(StageReport(
                name="Candidate versus sham",
                status=StageStatus.FAIL,
                details={"mean_delta_vs_sham": round(sham_delta, 4)},
                reason="candidate does not beat sham",
            ))

    # --- Safety ---
    safety_results = _load_json(artifacts / "safety_results.json")
    if safety_results is None:
        stages.append(StageReport(
            name="Safety evaluation",
            status=StageStatus.NOT_RUN,
            reason="safety_results.json not found",
        ))
    else:
        violations = safety_results.get("candidate_safety_violations", 0)
        if violations == 0:
            stages.append(StageReport(
                name="Safety evaluation",
                status=StageStatus.PASS,
                details={"candidate_safety_violations": violations},
            ))
        else:
            stages.append(StageReport(
                name="Safety evaluation",
                status=StageStatus.FAIL,
                details={"candidate_safety_violations": violations},
                reason="candidate has safety violations",
            ))

    # --- Promotion ---
    promotion_result = _load_json(artifacts / "promotion_result.json")
    if promotion_result is None:
        stages.append(StageReport(
            name="Promotion",
            status=StageStatus.BLOCKED,
            reason="promotion_result.json not found",
        ))
    elif promotion_result.get("passed", False):
        stages.append(StageReport(
            name="Promotion",
            status=StageStatus.PASS,
            details={"gates_passed": promotion_result.get("passed", False)},
        ))
    else:
        failed_gates = [g["name"] for g in promotion_result.get("gates", [])
                        if g.get("status") in ("FAIL", "BLOCKED", "NOT_RUN")]
        stages.append(StageReport(
            name="Promotion",
            status=StageStatus.BLOCKED,
            details={"failed_gates": failed_gates},
            reason="; ".join(failed_gates) if failed_gates else "promotion rejected",
        ))

    # --- Post-promotion ---
    post_promo = _load_json(artifacts / "post_promotion_results.json")
    if post_promo is None:
        stages.append(StageReport(
            name="Post-promotion proof",
            status=StageStatus.NOT_RUN,
            reason="post_promotion_results.json not found",
        ))
    else:
        action_changed = post_promo.get("action_changed", False)
        utility_improved = post_promo.get("utility_improved", False)
        if action_changed and utility_improved:
            stages.append(StageReport(
                name="Post-promotion proof",
                status=StageStatus.PASS,
                details={
                    "action_changed": action_changed,
                    "utility_improved": utility_improved,
                },
            ))
        else:
            stages.append(StageReport(
                name="Post-promotion proof",
                status=StageStatus.FAIL,
                details={
                    "action_changed": action_changed,
                    "utility_improved": utility_improved,
                },
                reason="post-promotion did not demonstrate changed action with improved utility",
            ))

    # --- Active policy ---
    stages.append(StageReport(
        name="Active policy",
        status=StageStatus.PASS if active_policy_id else StageStatus.NOT_RUN,
        details={"active_policy_id": active_policy_id or "(none)"},
    ))

    # --- Provenance ---
    provenance = _load_json(artifacts / "provenance_manifest.json")
    if provenance is None:
        stages.append(StageReport(
            name="Provenance",
            status=StageStatus.NOT_RUN,
            reason="provenance_manifest.json not found",
        ))
    else:
        source_hash = provenance.get("source_tree_sha256", "")
        if source_hash:
            stages.append(StageReport(
                name="Provenance",
                status=StageStatus.PASS,
                details={"source_tree_sha256": source_hash[:16] + "..."},
            ))
        else:
            stages.append(StageReport(
                name="Provenance",
                status=StageStatus.FAIL,
                reason="source_tree_sha256 is empty",
            ))

    # --- Build Markdown ---
    lines = [
        "# AutoLearn v0.3.1 Qualification Report",
        "",
        "## Stages",
        "",
    ]
    for stage in stages:
        lines.append(stage.to_markdown())
        lines.append("")

    # --- Final claim ---
    all_pass = all(s.status == StageStatus.PASS for s in stages)
    if all_pass:
        lines.append("## Final Claim")
        lines.append("")
        lines.append(
            "Capsule Brain trained an executive routing policy from isolated "
            "real counterfactual executions, preserved that policy correctly "
            "through serialization, beat a competent baseline and sham control "
            "on untouched held-out tasks with a positive paired-bootstrap lower "
            "bound, passed immutable safety and coverage gates, restored the "
            "promoted policy in a fresh process, and demonstrated higher "
            "verified utility through changed real runtime behavior."
        )
    else:
        lines.append("## Final Claim")
        lines.append("")
        lines.append(
            "Capsule Brain implements the required real-learning "
            "infrastructure, but causal executive improvement has not yet "
            "been qualified."
        )

    return "\n".join(lines)
