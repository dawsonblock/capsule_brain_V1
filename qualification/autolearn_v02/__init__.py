"""AutoLearn v0.2 qualification harness.

End-to-end pipeline with real runtime integration:
  tasks.py              <- 400+ deterministic, machine-verifiable routing tasks
  run_counterfactuals.py <- real + simulated counterfactual execution
  build_dataset.py      <- weighted supervised dataset + archetype splits
  train_policy.py       <- multinomial logistic regression
  evaluate_policy.py    <- paired evaluation on held-out + OOD archetypes
  run_shadow_eval.py    <- real shadow mode with disagreement logging
  promote_policy.py     <- 11-gate promotion gate
  report.py             <- consolidated qualification report

v0.2 differences from v0.1:
- 400+ tasks across 25+ coding archetypes (no label leakage in features)
- Archetype-based train/test/OOD splits (genuine OOD = unseen archetypes)
- Real counterfactual execution via RealCounterfactualRuntime
- 11-gate promotion gate (adds workflow routing, over-routing, calibration)
- Calibration metrics (Brier, ECE) reported alongside the gate
- Distribution-shift guard v2 (PSI, unseen-value, tool-inventory, model-ID)
- SHA-256 provenance hashes on every artifact
"""
