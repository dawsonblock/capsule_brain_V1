"""AutoLearn v0.3.1 qualification harness.

This is the canonical qualification directory for Capsule Brain 2.15.2 /
AutoLearn v0.3.1. It implements the real causal qualification pipeline:

    real counterfactual results
      → real ExecutiveExperience rows
      → training dataset
      → candidate training
      → real validation
      → real test
      → real OOD
      → promotion
      → real post-promotion test

Official qualification defaults to real runtime. Simulated runtime is
allowed only for development and is labeled NOT QUALIFYING.
"""
