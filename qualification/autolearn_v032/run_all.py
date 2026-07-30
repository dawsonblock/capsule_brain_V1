"""End-to-end v0.3.2 qualification pipeline.

Runs all stages in order, stopping at the first hard failure. Each stage
produces its artifact; the final report is derived from the artifact tree.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from .build_benchmark import build_benchmark
from .build_dataset import build_dataset
from .build_runtime import build_real_qualification_runtime
from .config import QualificationConfig
from .provenance import build_provenance
from .report import build_report
from .run_counterfactuals import run_counterfactuals
from .run_post_promotion import run_post_promotion
from .run_promotion import run_promotion
from .schemas import write_json
from .train_candidate import train_candidate
from .train_sham import train_sham


async def run_pipeline(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    def stage(name: str, data: dict) -> None:
        log.append({"stage": name, **data})

    # 1. Build benchmark
    try:
        _, bm, sm = build_benchmark(config)
        write_json(artifacts / "benchmark_manifest.json", bm)
        write_json(artifacts / "split_manifest.json", sm)
        stage("build_benchmark", {"status": "ok", "n_tasks": bm["total_tasks"]})
    except Exception as exc:
        stage("build_benchmark", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 2. Build runtime (proves real bootstrap path)
    try:
        runtime = build_real_qualification_runtime(config)
        stage("build_runtime", {"status": "ok", "runtime_type": runtime.runtime_type})
    except Exception as exc:
        stage("build_runtime", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 3. Run counterfactuals
    try:
        rows = await run_counterfactuals(config)
        stage("run_counterfactuals", {"status": "ok", "n_rows": len(rows), "verified": sum(1 for r in rows if r["verified_success"])})
    except Exception as exc:
        stage("run_counterfactuals", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 4. Build dataset
    try:
        ds = build_dataset(config)
        stage("build_dataset", {"status": "ok", "n_examples": ds["total_examples"]})
    except Exception as exc:
        stage("build_dataset", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 5. Train candidate
    try:
        _, train_record, _ = train_candidate(config)
        stage("train_candidate", {"status": "ok", "val_acc": train_record["validation_accuracy"]})
    except Exception as exc:
        stage("train_candidate", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 6. Train sham
    try:
        _, sham_record, _ = train_sham(config)
        stage("train_sham", {"status": "ok", "val_acc": sham_record["validation_accuracy"]})
    except Exception as exc:
        stage("train_sham", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 7. Promotion (runs all evals internally)
    try:
        promo = run_promotion(config)
        stage("run_promotion", {"status": "ok", "approved": promo["promotion_approved"], "n_passed": promo["n_passed"]})
    except Exception as exc:
        stage("run_promotion", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 8. Post-promotion
    try:
        post = run_post_promotion(config)
        stage("run_post_promotion", {"status": "ok", "passes": post["passes"]})
    except Exception as exc:
        stage("run_post_promotion", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 9. Provenance
    try:
        prov = build_provenance(config)
        stage("provenance", {"status": "ok", "final_hash": prov["final_hash"][:16]})
    except Exception as exc:
        stage("provenance", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    # 10. Report
    try:
        report = build_report(config)
        stage("report", {"status": "ok", "result": report["status"]})
    except Exception as exc:
        stage("report", {"status": "error", "reason": str(exc)})
        return {"status": "NOT RUN", "log": log}

    result = {
        "status": report["status"],
        "log": log,
        "report": report,
    }
    write_json(artifacts / "pipeline_log.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full v0.3.2 qualification pipeline.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="run only smoke subset")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()

    config = QualificationConfig(
        runtime=args.runtime,
        smoke=args.smoke,
        artifacts_dir=args.artifacts_dir,
    )
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)

    result = asyncio.run(run_pipeline(config))
    print(f"pipeline: {result['status']}")
    for entry in result["log"]:
        print(f"  {entry['stage']}: {entry.get('status', '')} {entry.get('reason', '')}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
