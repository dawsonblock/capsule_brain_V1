"""AutoLearn v0.4.2: Routing-Headroom Analysis, Sham-Control Hardening, and Gate A Identifiability Repair.

This subpackage contains diagnostic modules that consume the existing
7B full-run counterfactual outcomes to answer why Gate A failed and
whether there is enough causal routing signal for a learned executive
policy to outperform a competent sham.

All diagnostics consume existing artifacts — no new model executions
unless a specific experiment requires them.
"""
