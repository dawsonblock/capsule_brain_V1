"""AutoLearn Qualification v0.4.7 — Evidence-Complete Causal Gate Hierarchy.

This module implements the staged gate hierarchy:

    Gate A0 — Evidence admissibility
    Gate A1 — Routing headroom
    Gate A2 — Candidate causal effectiveness
    Gate A3 — Robustness and replication
    Gate A4 — Promotion eligibility

Key scientific rule: Gate A0 PASS does NOT imply Gate A effectiveness.
The candidate must beat both baseline and sham with LCB exceeding the
practical-effect threshold.
"""
from __future__ import annotations

PROTOCOL_VERSION = "0.4.7"
QUALIFICATION_VERSION = "0.4.7"
