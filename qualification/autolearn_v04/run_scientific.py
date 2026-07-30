"""Scientific qualification pipeline runner using Modal GPU (Section 12/17).

Runs the full scientific qualification on a real frozen transformer model:
    1. Build scientific benchmark (hidden answers)
    2. Execute counterfactuals on Modal GPU
    3. Diagnose runtime completion
    4. Build dataset from real model outcomes
    5. Train candidate and sham policies
    6. Evaluate Gate A (candidate vs baseline vs sham vs oracle)
    7. Collect hidden states for Gate B
    8. Evaluate Gate B (passive latent signal)
    9. Generate promotion decision and report

Usage:
    python -m qualification.autolearn_v04.run_scientific \\
        --artifacts-dir /tmp/scientific_qual \\
        --n-experience 50 --n-validation 20 --n-test 30 \\
        --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Import scientific benchmark builder.
from .scientific_benchmark import build_scientific_benchmark
from .schemas import sha256_json, sha256_text, write_json


def run_scientific_pipeline(
    artifacts_dir: str = "/tmp/scientific_qual",
    n_experience: int = 50,
    n_validation: int = 20,
    n_test: int = 30,
    n_ood: int = 20,
    n_safety: int = 10,
    crossover_fraction: float = 0.25,
    task_seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 256,
    collect_hidden_states: bool = True,
    hidden_layer_ids: list[int] | None = None,
) -> dict:
    """Run the full scientific qualification pipeline on Modal GPU."""
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if hidden_layer_ids is None:
        # Qwen2.5-3B has 36 layers. Collect from several points.
        hidden_layer_ids = [0, 12, 24, 35]

    print("=" * 70)
    print("CAPSULE BRAIN 2.15.4 / AutoLearn 0.4.0 SCIENTIFIC QUALIFICATION")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Tasks: exp={n_experience} val={n_validation} test={n_test} ood={n_ood} safety={n_safety}")
    print(f"Hidden states: {collect_hidden_states} layers={hidden_layer_ids}")
    print()

    # 1. Build scientific benchmark.
    print("[1/9] Building scientific benchmark...")
    tasks = build_scientific_benchmark(
        n_experience=n_experience,
        n_validation=n_validation,
        n_test=n_test,
        n_ood=n_ood,
        n_safety=n_safety,
        crossover_fraction=crossover_fraction,
        task_seed=task_seed,
    )
    # Build manifest.
    task_digests = [sha256_text(json.dumps(t.to_dict() if hasattr(t, 'to_dict') else t, sort_keys=True)) for t in tasks]
    benchmark_manifest = {
        "schema_version": "scientific-benchmark/1",
        "package_version": "2.15.4",
        "qualification_version": "0.4.0",
        "model_id": model_id,
        "provider_class": "real_model",
        "n_tasks": len(tasks),
        "n_experience": n_experience,
        "n_validation": n_validation,
        "n_test": n_test,
        "n_ood": n_ood,
        "n_safety": n_safety,
        "crossover_fraction": crossover_fraction,
        "task_seed": task_seed,
        "benchmark_digest": sha256_json({"task_digests": task_digests}),
        "tasks": [t.to_dict() if hasattr(t, 'to_dict') else t for t in tasks],
    }
    write_json(artifacts / "benchmark_manifest.json", benchmark_manifest)
    print(f"  Built {len(tasks)} scientific tasks")

    # Build split manifest.
    split_counts = {}
    for t in tasks:
        t_dict = t.to_dict() if hasattr(t, 'to_dict') else t
        split_counts[t_dict["split"]] = split_counts.get(t_dict["split"], 0) + 1
    split_manifest = {
        "schema_version": "split-manifest/1",
        "splits": split_counts,
        "split_grouping": "strict",
        "crossover_fraction": crossover_fraction,
    }
    write_json(artifacts / "split_manifest.json", split_manifest)

    # 2. Execute counterfactuals on Modal GPU.
    print("\n[2/9] Executing counterfactuals on Modal GPU...")
    from .modal_scientific_executor import run_scientific_counterfactuals_to_artifacts
    outcomes = run_scientific_counterfactuals_to_artifacts(
        config=None,  # not needed for scientific executor
        artifacts_dir=artifacts,
        tasks=tasks,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        collect_hidden_states=collect_hidden_states,
        hidden_layer_ids=hidden_layer_ids,
    )

    # 3. Runtime completion diagnostics.
    print("\n[3/9] Runtime completion diagnostics...")
    from .diagnose_runtime_completion import diagnose_runtime_completion
    from .config import QualificationConfig
    config = QualificationConfig(mode="scientific", runtime="real", artifacts_dir=str(artifacts))
    diag = diagnose_runtime_completion(config)
    print(f"  Status: {diag['status']}")
    for action, rate in diag.get("completion_rates", {}).items():
        print(f"    {action}: {rate:.1%}")

    # 4. Build dataset from real model outcomes.
    print("\n[4/9] Building dataset from real model outcomes...")
    from .build_dataset import build_dataset
    try:
        build_dataset(config)
        print("  Dataset built")
    except Exception as e:
        print(f"  WARNING: Dataset build issue: {e}")

    # 5. Train candidate and sham policies.
    print("\n[5/9] Training candidate policy...")
    from .train_candidate import train_candidate
    try:
        candidate_policy, candidate_training, candidate_prov = train_candidate(config)
        print(f"  Candidate trained: {candidate_policy.policy_id}")
    except Exception as e:
        print(f"  WARNING: Candidate training issue: {e}")

    print("\n[6/9] Training sham policy...")
    from .train_sham import train_sham
    try:
        sham_policy, sham_training = train_sham(config)
        print(f"  Sham trained: {sham_policy.policy_id}")
    except Exception as e:
        print(f"  WARNING: Sham training issue: {e}")

    # 6. Gate A evaluation.
    print("\n[7/9] Gate A evaluation (candidate vs baseline vs sham vs oracle)...")
    from .evaluate_gate_a import evaluate_gate_a
    try:
        gate_a_result = evaluate_gate_a(config)
        print(f"  Gate A status: {gate_a_result.get('status', 'UNKNOWN')}")
        if "delta_p1_p0" in gate_a_result:
            print(f"  P1-P0 mean delta: {gate_a_result['delta_p1_p0']['mean']:.4f}")
    except Exception as e:
        print(f"  WARNING: Gate A issue: {e}")

    # 7. Gate B evaluation (if hidden states collected).
    gate_b_result = None
    if collect_hidden_states:
        print("\n[8/9] Gate B evaluation (passive latent signal)...")
        from .evaluate_gate_b import evaluate_gate_b
        try:
            gate_b_result = evaluate_gate_b(config)
            print(f"  Gate B status: {gate_b_result.get('status', 'UNKNOWN')}")
        except Exception as e:
            print(f"  WARNING: Gate B issue: {e}")

    # 8. Promotion and report.
    print("\n[9/9] Promotion decision and report...")
    from .run_promotion import run_promotion
    try:
        promo = run_promotion(config)
        print(f"  PROMOTION_DECISION: {promo.get('PROMOTION_DECISION', 'BLOCKED')}")
        print(f"  Gate A: {promo.get('gate_a_status', 'NOT_RUN')}")
        print(f"  Gate B: {promo.get('gate_b_status', 'NOT_RUN')}")
    except Exception as e:
        print(f"  WARNING: Promotion issue: {e}")

    # Generate report.
    from .report import generate_report
    try:
        generate_report(config)
    except Exception as e:
        print(f"  WARNING: Report issue: {e}")

    elapsed = time.time() - start_time

    # Final summary.
    print("\n" + "=" * 70)
    print("SCIENTIFIC QUALIFICATION SUMMARY")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Total outcomes: {len(outcomes)}")
    n_success = sum(1 for o in outcomes if o.get("verification") == "success")
    print(f"Verified success: {n_success}/{len(outcomes)} ({n_success/len(outcomes)*100:.1f}%)")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Artifacts: {artifacts_dir}")

    return {
        "model_id": model_id,
        "n_tasks": len(tasks),
        "n_outcomes": len(outcomes),
        "n_success": n_success,
        "elapsed_seconds": elapsed,
        "artifacts_dir": str(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scientific qualification on Modal GPU.")
    parser.add_argument("--artifacts-dir", default="/tmp/scientific_qual")
    parser.add_argument("--n-experience", type=int, default=50)
    parser.add_argument("--n-validation", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--n-ood", type=int, default=20)
    parser.add_argument("--n-safety", type=int, default=10)
    parser.add_argument("--crossover-fraction", type=float, default=0.25)
    parser.add_argument("--task-seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-hidden-states", action="store_true")
    parser.add_argument("--hidden-layers", type=str, default="0,12,24,35",
                        help="Comma-separated layer indices for Gate B")
    args = parser.parse_args()

    hidden_layer_ids = [int(x) for x in args.hidden_layers.split(",")]

    result = run_scientific_pipeline(
        artifacts_dir=args.artifacts_dir,
        n_experience=args.n_experience,
        n_validation=args.n_validation,
        n_test=args.n_test,
        n_ood=args.n_ood,
        n_safety=args.n_safety,
        crossover_fraction=args.crossover_fraction,
        task_seed=args.task_seed,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        collect_hidden_states=not args.no_hidden_states,
        hidden_layer_ids=hidden_layer_ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
