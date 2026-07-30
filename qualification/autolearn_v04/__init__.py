"""Capsule Brain AutoLearn v0.4.0 official qualification package.

This is the active, scientifically-honest qualification pipeline for
Capsule Brain 2.15.4 / AutoLearn 0.3.3 / Qualification protocol v0.4.0.

It supersedes v0.3.2 and answers two questions correctly:

  Gate A -- Causal AutoLearn:
      Does a candidate executive policy trained from real counterfactual
      experience improve held-out verified utility over both the frozen
      baseline and a sham-learning control?

  Gate B -- Passive latent verification (CLUE):
      Do hidden-state activation deltas predict independently verified
      success on held-out trajectories better than chance and stronger
      non-latent confidence baselines?

Hard guarantees introduced in v0.4.0:

  * Missing counterfactual executions are NEVER scored as catastrophic
    failures. They are represented by typed OutcomeAvailability states and
    receive NO behavioral utility.
  * Smoke mode never issues a scientific promotion verdict. It reports
    PIPELINE_INTEGRITY only; SCIENTIFIC_QUALIFICATION is NOT_RUN and
    PROMOTION_DECISION is BLOCKED.
  * Every trained policy carries a complete PolicyProvenance manifest with
    non-empty SHA-256 digests over source tree, benchmark, split,
    counterfactual results, dataset, feature schema, hyperparameters,
    learner code, runtime config, verifier config, model identity, and
    tokenizer identity. The empty-content SHA-256 is rejected.
  * Promotion independently recomputes and compares every provenance digest.
  * Primary evidence gates are counted separately from the summary decision.
  * The deterministic grounded provider is relabeled "runtime integration
    provider" and may only qualify infrastructure -- never a causal
    learning claim. A real model backend is required for Gate A.
  * Gate B (CLUE) is passive: it reads hidden states without altering
    generation.

The chain is:

    build_benchmark
      -> run_counterfactuals (real runtime, isolated per-action services,
         typed OutcomeAvailability, no -1000 placeholders)
      -> build_dataset (counterfactual-completeness-filtered)
      -> train_candidate (D_experience only, trust-region constrained)
      -> train_sham (permuted labels within valid constraints)
      -> evaluate_gate_a (baseline / candidate / sham / oracle on held-out
         test under complete action coverage, grouped bootstrap)
      -> collect_activations (passive hidden-state extraction)
      -> evaluate_gate_b (CLUE prototypes vs. non-latent baselines)
      -> run_promotion (gate accounting with GateStatus, independent
         provenance verification)
      -> run_post_promotion (fresh-process behavioral proof)
      -> provenance (full lineage digest)
      -> report (honest claim language)

If any required stage is missing, incomplete, or run on the infrastructure-
only provider for a causal claim, the report must say BLOCKED / NOT RUN,
never PASS.
"""
from capsule_brain.version import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
    PROTOCOL_VERSION,
)
