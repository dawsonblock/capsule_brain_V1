"""Capsule Brain AutoLearn v0.3.2 official qualification package.

This is the active, executable qualification pipeline for Capsule Brain
2.15.3 / AutoLearn v0.3.2. Every stage implements its named function — no
placeholder may return success without producing or validating its artifact.

The chain is:

    build_benchmark
      -> run_counterfactuals (real runtime, isolated per-action services)
      -> build_dataset (real ExecutiveExperience rows only)
      -> train_candidate (D_experience only)
      -> train_sham
      -> evaluate_validation / evaluate_test / evaluate_ood / evaluate_oracle
         / evaluate_safety
      -> run_promotion (25 mandatory gates)
      -> run_post_promotion (fresh process behavioral proof)
      -> provenance
      -> report

If any required stage is missing or simulated, the report must say
BLOCKED / NOT RUN, never PASS.
"""
AUTOLEARN_QUALIFICATION_VERSION = "0.3.2"
PACKAGE_VERSION = "2.15.3"
