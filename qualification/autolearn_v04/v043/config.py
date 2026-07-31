"""Configuration constants for v0.4.3 repair analysis."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
PACKAGE_VERSION = "2.15.10"
AUTOLEARN_VERSION = "0.3.8"
QUALIFICATION_VERSION = "0.4.6"

# ---------------------------------------------------------------------------
# Recovered-headroom calculation
# ---------------------------------------------------------------------------
MINIMUM_DENOMINATOR = 1e-10
HEADROOM_PERCENTAGE_SCALE = 100.0

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Sham ensemble
# ---------------------------------------------------------------------------
SHAM_N_SEEDS = 50
SHAM_BASE_SEED = 10000

# ---------------------------------------------------------------------------
# Candidate stability
# ---------------------------------------------------------------------------
CANDIDATE_N_SEEDS = 50
CANDIDATE_BASE_SEED = 20000

# ---------------------------------------------------------------------------
# Gate A thresholds
# ---------------------------------------------------------------------------
# A1: Routing headroom
A1_HEADROOM_THRESHOLD = 0.05
A1_CONCENTRATION_HHI_THRESHOLD = 0.25  # max HHI for "broad"

# A3: Candidate vs baseline
A3_PRACTICAL_THRESHOLD = 0.01

# A4: Candidate vs sham
A4_PRACTICAL_THRESHOLD = 0.0
A4_SHAM_PERCENTILE_THRESHOLD = 50  # candidate must beat >50% of shams

# A5: High-margin recovery
A5_MIN_RECOVERY_FRACTION = 0.10
A5_MARGIN_BUCKETS = ["MEANINGFUL", "HIGH_VALUE"]

# ---------------------------------------------------------------------------
# Scale decision
# ---------------------------------------------------------------------------
SCALE_HEADROOM_THRESHOLD = 0.05
SCALE_FEATURE_SIGNAL_REQUIRED = True
SCALE_STABILITY_REQUIRED = True

# ---------------------------------------------------------------------------
# Matched 14B pilot
# ---------------------------------------------------------------------------
PILOT_MIN_TASKS = 20
PILOT_MAX_TASKS = 30

# ---------------------------------------------------------------------------
# Metric consistency tolerance
# ---------------------------------------------------------------------------
METRIC_TOLERANCE_STRICT = 1e-9
METRIC_TOLERANCE_FLOAT = 1e-6

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
UTILITY_VERSION = "normalized_v1"
