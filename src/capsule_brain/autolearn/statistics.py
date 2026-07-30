"""Statistical tests for AutoLearn v2.16.0 qualification.

Priority 5: Replaces the fake bootstrap interval (normal approximation) with
actual paired bootstrap, paired permutation test, and sign test.

- ``paired_bootstrap``: 10,000 bootstrap resamples of paired deltas.
  Reports mean, median, 95% BCa interval, and standard error.
- ``paired_permutation_test``: Exact or sampled permutation test for
  H0: no treatment effect.
- ``sign_test``: Non-parametric sign test for consistent positive direction.

Priority 6: Minimum effect threshold. Qualification requires
``lower_95_ci > epsilon``, not merely ``> 0``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BootstrapResult:
    """Result of a paired bootstrap analysis."""
    mean: float
    median: float
    std_error: float
    ci_low: float  # 95% BCa lower bound
    ci_high: float  # 95% BCa upper bound
    n_bootstrap: int
    n_samples: int
    epsilon: float = 0.0  # minimum effect threshold
    passes_threshold: bool = False  # True if ci_low > epsilon
    bootstrap_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 8),
            "median": round(self.median, 8),
            "std_error": round(self.std_error, 8),
            "ci_low": round(self.ci_low, 8),
            "ci_high": round(self.ci_high, 8),
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
            "epsilon": round(self.epsilon, 8),
            "passes_threshold": self.passes_threshold,
        }


@dataclass(slots=True)
class PermutationTestResult:
    """Result of a paired permutation test."""
    observed_statistic: float
    p_value: float
    n_permutations: int
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_statistic": round(self.observed_statistic, 8),
            "p_value": round(self.p_value, 8),
            "n_permutations": self.n_permutations,
            "n_samples": self.n_samples,
        }


@dataclass(slots=True)
class SignTestResult:
    """Result of a non-parametric sign test."""
    n_positive: int
    n_negative: int
    n_zero: int
    p_value: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_zero": self.n_zero,
            "p_value": round(self.p_value, 8),
            "n_samples": self.n_samples,
        }


@dataclass(slots=True)
class QualificationStatistics:
    """Complete statistical analysis for qualification."""
    bootstrap: BootstrapResult
    permutation: PermutationTestResult
    sign_test: SignTestResult
    deltas: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap": self.bootstrap.to_dict(),
            "permutation": self.permutation.to_dict(),
            "sign_test": self.sign_test.to_dict(),
            "n_samples": len(self.deltas),
            "mean_delta": round(sum(self.deltas) / len(self.deltas), 8) if self.deltas else 0.0,
        }


def paired_bootstrap(
    deltas: list[float],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    epsilon: float = 0.0,
    seed: int = 42,
) -> BootstrapResult:
    """Compute paired bootstrap confidence interval with BCa correction.

    Args:
        deltas: Paired differences Δ_i = U(candidate_i) - U(baseline_i).
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        confidence: Confidence level (default 0.95).
        epsilon: Minimum effect threshold. ``passes_threshold`` is True
            iff ``ci_low > epsilon``.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with mean, median, 95% BCa interval, std error.
    """
    if not deltas:
        return BootstrapResult(
            mean=0.0, median=0.0, std_error=0.0,
            ci_low=0.0, ci_high=0.0,
            n_bootstrap=n_bootstrap, n_samples=0,
            epsilon=epsilon, passes_threshold=False,
        )

    n = len(deltas)
    rng = random.Random(seed)
    observed_mean = sum(deltas) / n

    # Generate bootstrap resamples.
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample = [deltas[i] for i in indices]
        boot_means.append(sum(sample) / n)

    boot_means.sort()

    # BCa correction.
    # Acceleration factor (a) using jackknife.
    jackknife_means = []
    for i in range(n):
        loo = deltas[:i] + deltas[i + 1:]
        jackknife_means.append(sum(loo) / (n - 1) if loo else 0.0)
    jack_mean = sum(jackknife_means) / n
    denom = sum((jm - jack_mean) ** 3 for jm in jackknife_means)
    numer = sum((jm - jack_mean) ** 2 for jm in jackknife_means) ** 1.5
    a = denom / (6 * numer) if numer > 0 else 0.0

    # Bias correction factor (z0).
    count_below = sum(1 for bm in boot_means if bm < observed_mean)
    # Use a safe proportion to avoid math domain errors.
    prop_below = count_below / n_bootstrap
    # Clamp to avoid 0 or 1 which produce ±inf in the inverse normal CDF.
    prop_below = max(min(prop_below, 1.0 - 1e-10), 1e-10)
    z0 = _inv_norm_cdf(prop_below)

    # BCa adjusted percentiles.
    alpha = 1.0 - confidence
    z_low = _inv_norm_cdf(alpha / 2)
    z_high = _inv_norm_cdf(1.0 - alpha / 2)

    a1 = _bca_adjust(z0, a, z_low)
    a2 = _bca_adjust(z0, a, z_high)

    idx_low = max(0, min(n_bootstrap - 1, int(a1 * n_bootstrap)))
    idx_high = max(0, min(n_bootstrap - 1, int(a2 * n_bootstrap)))

    ci_low = boot_means[idx_low]
    ci_high = boot_means[idx_high]

    median = boot_means[n_bootstrap // 2]
    std_error = math.sqrt(
        sum((bm - observed_mean) ** 2 for bm in boot_means) / n_bootstrap
    )

    passes = ci_low > epsilon

    return BootstrapResult(
        mean=observed_mean,
        median=median,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        epsilon=epsilon,
        passes_threshold=passes,
        bootstrap_samples=boot_means,
    )


def _bca_adjust(z0: float, a: float, z: float) -> float:
    """BCa percentile adjustment."""
    denom = 1.0 - a * (z0 + z)
    if abs(denom) < 1e-10:
        return 0.5
    adjusted = _norm_cdf(z0 + (z0 + z) / denom)
    return max(0.0, min(1.0, adjusted))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Uses the Acklam algorithm for reasonable accuracy.
    """
    # Coefficients for the Acklam approximation.
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def paired_permutation_test(
    deltas: list[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> PermutationTestResult:
    """Paired permutation test for H0: no treatment effect.

    Under H0, the sign of each Δ_i is equally likely to be positive or
    negative. We randomly flip signs and compute the mean to build the
    null distribution.

    Args:
        deltas: Paired differences Δ_i = U(candidate_i) - U(baseline_i).
        n_permutations: Number of permutations (default 10,000).
        seed: Random seed.

    Returns:
        PermutationTestResult with observed statistic and p-value.
    """
    if not deltas:
        return PermutationTestResult(
            observed_statistic=0.0, p_value=1.0,
            n_permutations=n_permutations, n_samples=0,
        )

    n = len(deltas)
    rng = random.Random(seed)
    observed = sum(deltas) / n

    count_extreme = 0
    for _ in range(n_permutations):
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)]
        perm_stat = sum(d * s for d, s in zip(deltas, signs)) / n
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        n_samples=n,
    )


def sign_test(deltas: list[float]) -> SignTestResult:
    """Non-parametric sign test for consistent positive direction.

    Tests whether the majority of deltas are positive (candidate > baseline).

    Args:
        deltas: Paired differences Δ_i = U(candidate_i) - U(baseline_i).

    Returns:
        SignTestResult with counts and p-value (two-sided binomial test).
    """
    if not deltas:
        return SignTestResult(
            n_positive=0, n_negative=0, n_zero=0,
            p_value=1.0, n_samples=0,
        )

    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_zero = sum(1 for d in deltas if d == 0)

    # Two-sided exact binomial test.
    n_nonzero = n_positive + n_negative
    if n_nonzero == 0:
        p_value = 1.0
    else:
        # p-value = 2 * min(P(X <= k), P(X >= k)) where X ~ Bin(n, 0.5)
        k = max(n_positive, n_negative)
        p_left = _binomial_cdf(k, n_nonzero, 0.5)
        p_right = 1.0 - _binomial_cdf(k - 1, n_nonzero, 0.5) if k > 0 else 1.0
        p_value = min(1.0, 2.0 * min(p_left, p_right))

    return SignTestResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        p_value=p_value,
        n_samples=len(deltas),
    )


def _binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial probability mass function P(X = k)."""
    if k < 0 or k > n:
        return 0.0
    # Use log to avoid overflow for large n.
    log_coeff = _log_binomial_coeff(n, k)
    log_prob = log_coeff + k * math.log(p) + (n - k) * math.log(1.0 - p)
    return math.exp(log_prob)


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF P(X <= k)."""
    return sum(_binomial_pmf(i, n, p) for i in range(k + 1))


def _log_binomial_coeff(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k)."""
    k = min(k, n - k)
    if k == 0:
        return 0.0
    result = 0.0
    for i in range(k):
        result += math.log(n - i) - math.log(i + 1)
    return result


def compute_qualification_statistics(
    candidate_utilities: list[float],
    baseline_utilities: list[float],
    *,
    n_bootstrap: int = 10_000,
    epsilon: float = 0.0,
    seed: int = 42,
) -> QualificationStatistics:
    """Compute complete statistical analysis for qualification.

    Args:
        candidate_utilities: Per-task utilities for the candidate policy.
        baseline_utilities: Per-task utilities for the baseline policy.
        n_bootstrap: Number of bootstrap resamples.
        epsilon: Minimum effect threshold (Priority 6).
        seed: Random seed for reproducibility.

    Returns:
        QualificationStatistics with bootstrap, permutation, and sign test.
    """
    if len(candidate_utilities) != len(baseline_utilities):
        raise ValueError(
            f"utility lists must have equal length: "
            f"candidate={len(candidate_utilities)}, baseline={len(baseline_utilities)}"
        )

    deltas = [c - b for c, b in zip(candidate_utilities, baseline_utilities)]

    boot = paired_bootstrap(
        deltas, n_bootstrap=n_bootstrap, epsilon=epsilon, seed=seed
    )
    perm = paired_permutation_test(deltas, n_permutations=n_bootstrap, seed=seed)
    st = sign_test(deltas)

    return QualificationStatistics(
        bootstrap=boot,
        permutation=perm,
        sign_test=st,
        deltas=deltas,
    )
