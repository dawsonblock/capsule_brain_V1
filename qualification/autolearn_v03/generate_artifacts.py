"""Generate remaining required qualification artifacts (v0.3).

Produces:
- baseline_results.json  (baseline metrics from test + ood)
- candidate_results.json (candidate metrics from test + ood)
- ood_results.json       (OOD-only evaluation details)
- calibration.json       (calibration metrics extracted from evaluation)
- executive_experiences.jsonl (all counterfactual experiences as JSONL)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.counterfactual import (
    SimulatedCounterfactualRuntime,
    run_counterfactuals,
)
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import build_all_tasks


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    test_eval = eval_data.get("test", {})
    ood_eval = eval_data.get("ood", {})
    v2 = test_eval.get("v2_metrics", {})
    ood_v2 = ood_eval.get("v2_metrics", {})

    # baseline_results.json
    baseline_results = {
        "test": test_eval.get("baseline_metrics", {}),
        "ood": ood_eval.get("baseline_metrics", {}),
    }
    text = json.dumps(baseline_results, sort_keys=True, indent=2)
    (out_dir / "baseline_results.json").write_text(text)
    print(f"wrote baseline_results.json (sha256={_compute_sha256(text)[:16]})")

    # candidate_results.json
    candidate_results = {
        "test": test_eval.get("candidate_metrics", {}),
        "ood": ood_eval.get("candidate_metrics", {}),
        "test_v2_metrics": v2,
        "ood_v2_metrics": ood_v2,
    }
    text = json.dumps(candidate_results, sort_keys=True, indent=2)
    (out_dir / "candidate_results.json").write_text(text)
    print(f"wrote candidate_results.json (sha256={_compute_sha256(text)[:16]})")

    # ood_results.json
    ood_results = {
        "baseline": ood_eval.get("baseline_metrics", {}),
        "candidate": ood_eval.get("candidate_metrics", {}),
        "mean_delta": ood_eval.get("mean_delta", 0),
        "delta_ci_low": ood_eval.get("delta_ci_low", 0),
        "delta_ci_high": ood_eval.get("delta_ci_high", 0),
        "wins": ood_eval.get("wins", 0),
        "ties": ood_eval.get("ties", 0),
        "losses": ood_eval.get("losses", 0),
        "v2_metrics": ood_v2,
    }
    text = json.dumps(ood_results, sort_keys=True, indent=2)
    (out_dir / "ood_results.json").write_text(text)
    print(f"wrote ood_results.json (sha256={_compute_sha256(text)[:16]})")

    # calibration.json
    cal_data = v2.get("calibration", {})
    text = json.dumps(cal_data, sort_keys=True, indent=2)
    (out_dir / "calibration.json").write_text(text)
    print(f"wrote calibration.json (sha256={_compute_sha256(text)[:16]})")

    # executive_experiences.jsonl — all counterfactual experiences
    # v2.16.0: use real runtime for official qualification.
    tasks = build_all_tasks()
    from capsule_brain.autolearn.counterfactual import RealCounterfactualRuntime
    from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory
    factory = QualificationServiceFactory()
    runtime = RealCounterfactualRuntime(service_factory=factory)
    rows = run_counterfactuals(tasks, runtime, utility_config=UtilityConfig())
    lines = []
    for r in rows:
        lines.append(json.dumps(r["experience"].to_dict(), sort_keys=True))
    text = "\n".join(lines) + "\n" if lines else ""
    (out_dir / "executive_experiences.jsonl").write_text(text)
    print(f"wrote executive_experiences.jsonl ({len(lines)} experiences, sha256={_compute_sha256(text)[:16]})")


if __name__ == "__main__":
    main()
