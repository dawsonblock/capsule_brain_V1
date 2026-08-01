"""Authoritative version constants for Capsule Brain.

All qualification code must import these constants. Do not duplicate
literal version strings in qualification packages.

Required invariant:
    installed package version
    == source package version
    == qualification report package version
    == provenance package version
"""
PACKAGE_VERSION = "2.15.11"
AUTOLEARN_VERSION = "0.3.10"
AUTOLEARN_QUALIFICATION_VERSION = "0.4.7"
PROTOCOL_VERSION = "0.4.7"

# Component-level versions (separate from the overall AutoLearn version).
AUTOLEARN_RUNTIME_CONTROLLER_VERSION = "0.2.3"
AUTOLEARN_LEARNER_VERSION = "0.3.10"
AUTOLEARN_POLICY_SCHEMA_VERSION = "0.3.1"
AUTOLEARN_FEATURE_SCHEMA_VERSION = "exec_features_v2"
