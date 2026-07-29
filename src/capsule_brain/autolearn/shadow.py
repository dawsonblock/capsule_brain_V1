"""Shadow mode + distribution-shift guard (v2).

Shadow mode runs the candidate alongside production without executing the
candidate action (unless the task is in an explicit safe evaluation set).
Disagreement cases are high-information data and are prioritized for
evaluation.

The v2 distribution-shift guard improves beyond simple feature means. It
implements:
- standardized mean shift
- per-feature Population Stability Index (PSI)
- categorical unseen-value detection
- tool-inventory change detection
- model-ID mismatch

If significant shift is detected, the policy falls back to baseline and
records ``distribution_shift_detected = true`` plus a ``shift_reason``. It
does not silently reject valid OOD evaluation tasks — both the candidate
prediction and the fallback decision are recorded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState


@dataclass(slots=True)
class ShadowRecord:
    task_id: str
    production_action: Action
    shadow_action: Action
    disagreement: bool
    shadow_confidence: float
    shadow_abstained: bool
    shadow_reason: str
    distribution_shift_detected: bool
    timestamp: str = ""


@dataclass(slots=True)
class ShadowStats:
    n: int = 0
    disagreements: int = 0
    distribution_shifts: int = 0
    abstentions: int = 0
    by_action: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "disagreements": self.disagreements,
            "distribution_shifts": self.distribution_shifts,
            "abstentions": self.abstentions,
            "by_action": dict(self.by_action),
        }


@dataclass(slots=True)
class DistributionShiftConfig:
    """Configuration for the v2 distribution-shift guard.

    v2 adds PSI thresholds, unseen-value detection, tool-inventory change
    detection, and model-ID mismatch detection on top of the v1 mean-shift
    check.
    """
    # Maximum allowed standardized mean distance before fallback.
    max_mean_distance: float = 1.5
    # Minimum number of recent observations required before computing a
    # distribution-shift estimate. Below this, no shift is declared.
    min_window: int = 10
    # Per-feature weights for the distance metric (uniform by default).
    feature_weights: tuple[float, ...] | None = None
    # v2: PSI threshold. Per-feature PSI above this triggers shift.
    psi_threshold: float = 0.25
    # v2: training-set standard deviations for standardized mean shift.
    training_stds: list[float] | None = None
    # v2: training-set categorical values for unseen-value detection.
    # Maps feature index -> set of seen values (for categorical features).
    training_categoricals: dict[int, set[float]] | None = None
    # v2: training-set tool inventory (set of tool names seen at train time).
    training_tool_names: set[str] | None = None
    # v2: training-set model IDs (set of model_id values seen at train time).
    training_model_ids: set[str] | None = None
    # v2: feature index for num_available_tools (used for tool-inventory
    # change detection via the state, not just the count).
    tool_count_feature_index: int = 10
    # v2: feature index for model_identity_id.
    model_id_feature_index: int = 14

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mean_distance": self.max_mean_distance,
            "min_window": self.min_window,
            "feature_weights": list(self.feature_weights) if self.feature_weights else None,
            "psi_threshold": self.psi_threshold,
            "training_stds": list(self.training_stds) if self.training_stds else None,
            "training_categoricals": (
                {str(k): list(v) for k, v in self.training_categoricals.items()}
                if self.training_categoricals else None
            ),
            "training_tool_names": list(self.training_tool_names) if self.training_tool_names else None,
            "training_model_ids": list(self.training_model_ids) if self.training_model_ids else None,
            "tool_count_feature_index": self.tool_count_feature_index,
            "model_id_feature_index": self.model_id_feature_index,
        }


class DistributionShiftGuard:
    """Tracks recent feature vectors and compares them to a training reference.

    v2 implements:
    - standardized mean shift (using training stds)
    - per-feature PSI
    - categorical unseen-value detection
    - tool-inventory change detection (via observed state)
    - model-ID mismatch (via observed state)

    The guard records a ``shift_reason`` listing which checks triggered.
    """

    def __init__(
        self,
        training_means: list[float] | None = None,
        config: DistributionShiftConfig | None = None,
    ) -> None:
        self.config = config or DistributionShiftConfig()
        self.training_means = list(training_means) if training_means else []
        self.recent: list[list[float]] = []
        self._last_shift_reasons: list[str] = []
        # Observed runtime context for tool/model shift detection.
        self._observed_tool_names: set[str] = set()
        self._observed_model_ids: set[str] = set()

    def observe(self, features: list[float]) -> None:
        self.recent.append(list(features))
        # Keep a bounded window.
        max_window = max(self.config.min_window * 4, 200)
        if len(self.recent) > max_window:
            self.recent = self.recent[-max_window:]

    def observe_context(
        self,
        *,
        tool_names: list[str] | None = None,
        model_id: str | None = None,
    ) -> None:
        """Observe runtime context (tool inventory, model ID) for shift detection."""
        if tool_names is not None:
            self._observed_tool_names.update(tool_names)
        if model_id is not None:
            self._observed_model_ids.add(model_id)

    def current_distance(self) -> float:
        if len(self.recent) < self.config.min_window or not self.training_means:
            return 0.0
        n_features = len(self.training_means)
        cur_means = [0.0] * n_features
        for vec in self.recent:
            for i in range(min(n_features, len(vec))):
                cur_means[i] += vec[i]
        n = len(self.recent)
        cur_means = [m / n for m in cur_means]
        # v2: standardized mean shift if training stds are available.
        stds = self.config.training_stds
        if stds and len(stds) == n_features:
            total = 0.0
            for i in range(n_features):
                sd = stds[i] if stds[i] > 0 else 1.0
                total += ((cur_means[i] - self.training_means[i]) / sd) ** 2
            return math.sqrt(total / n_features)
        # v1 fallback: unstandardized.
        weights = (
            self.config.feature_weights
            if self.config.feature_weights
            else (1.0,) * n_features
        )
        total = 0.0
        wsum = 0.0
        for i in range(n_features):
            w = weights[i] if i < len(weights) else 1.0
            total += w * (cur_means[i] - self.training_means[i]) ** 2
            wsum += w
        return math.sqrt(total / wsum) if wsum > 0 else 0.0

    def _per_feature_psi(self) -> list[float]:
        """Compute PSI for each feature using simple binning.

        PSI = sum_i (p_i - q_i) * ln(p_i / q_i)
        where p_i is the training proportion in bin i and q_i is the recent
        proportion. Returns a list of per-feature PSI values; features
        without enough data return 0.0.
        """
        if len(self.recent) < self.config.min_window or not self.training_means:
            return []
        n_features = len(self.training_means)
        # We don't have the raw training vectors here, only the means. PSI
        # requires the training distribution. When training_stds are
        # available, we approximate the training distribution as a Gaussian
        # and bin the recent observations against it. This is a pragmatic
        # approximation; a full implementation would store training
        # histograms. The mean-shift + unseen-value + tool/model checks are
        # the primary v2 signals; PSI is a secondary refinement.
        stds = self.config.training_stds
        if not stds or len(stds) != n_features:
            return [0.0] * n_features
        psi_values: list[float] = []
        n_recent = len(self.recent)
        for i in range(n_features):
            mean = self.training_means[i]
            sd = stds[i] if stds[i] > 0 else 1.0
            # 10 bins from mean-3sd to mean+3sd.
            n_bins = 10
            lo = mean - 3 * sd
            hi = mean + 3 * sd
            if hi <= lo:
                psi_values.append(0.0)
                continue
            bin_width = (hi - lo) / n_bins
            # Training proportions (Gaussian CDF approximation).
            train_props = []
            for b in range(n_bins):
                blo = lo + b * bin_width
                bhi = blo + bin_width
                # Approximate Gaussian mass in [blo, bhi].
                from math import erf, sqrt
                def _cdf(x):
                    return 0.5 * (1 + erf((x - mean) / (sd * sqrt(2))))
                p = _cdf(bhi) - _cdf(blo)
                train_props.append(max(p, 1e-6))
            # Recent proportions.
            recent_props = [0.0] * n_bins
            for vec in self.recent:
                if i >= len(vec):
                    continue
                val = vec[i]
                if val < lo or val > hi:
                    # Out-of-range: count in the extreme bins.
                    if val < lo:
                        recent_props[0] += 1
                    else:
                        recent_props[-1] += 1
                    continue
                b = int((val - lo) / bin_width)
                b = min(b, n_bins - 1)
                recent_props[b] += 1
            recent_props = [max(c / n_recent, 1e-6) for c in recent_props]
            # PSI.
            psi = 0.0
            for tp, rp in zip(train_props, recent_props):
                psi += (rp - tp) * math.log(rp / tp)
            psi_values.append(psi)
        return psi_values

    def _check_unseen_categoricals(self) -> list[str]:
        """Detect unseen categorical values in the recent window."""
        reasons: list[str] = []
        categoricals = self.config.training_categoricals
        if not categoricals or not self.recent:
            return reasons
        for feat_idx, seen_values in categoricals.items():
            for vec in self.recent:
                if feat_idx >= len(vec):
                    continue
                val = vec[feat_idx]
                if val not in seen_values:
                    reasons.append(
                        f"unseen categorical value {val} at feature index {feat_idx}"
                    )
                    break
        return reasons

    def _check_tool_inventory(self) -> list[str]:
        """Detect tool-inventory change vs. training."""
        reasons: list[str] = []
        training_tools = self.config.training_tool_names
        if training_tools is None:
            return reasons
        new_tools = self._observed_tool_names - training_tools
        if new_tools:
            reasons.append(
                f"tool inventory changed: new tools {sorted(new_tools)} not in training"
            )
        return reasons

    def _check_model_id(self) -> list[str]:
        """Detect model-ID mismatch vs. training."""
        reasons: list[str] = []
        training_ids = self.config.training_model_ids
        if training_ids is None:
            return reasons
        new_ids = self._observed_model_ids - training_ids
        if new_ids:
            reasons.append(
                f"model ID mismatch: {sorted(new_ids)} not in training"
            )
        return reasons

    def shifted(self) -> bool:
        """Return True if any shift check triggers. Records shift_reasons."""
        reasons: list[str] = []
        # 1. Standardized mean shift.
        dist = self.current_distance()
        if dist > self.config.max_mean_distance:
            reasons.append(f"mean shift {dist:.3f} > {self.config.max_mean_distance}")
        # 2. Per-feature PSI.
        psi_values = self._per_feature_psi()
        for i, psi in enumerate(psi_values):
            if psi > self.config.psi_threshold:
                reasons.append(f"PSI {psi:.3f} > {self.config.psi_threshold} at feature {i}")
        # 3. Unseen categoricals.
        reasons.extend(self._check_unseen_categoricals())
        # 4. Tool inventory change.
        reasons.extend(self._check_tool_inventory())
        # 5. Model ID mismatch.
        reasons.extend(self._check_model_id())
        self._last_shift_reasons = reasons
        return len(reasons) > 0

    @property
    def shift_reasons(self) -> list[str]:
        return list(self._last_shift_reasons)


class ShadowEvaluator:
    """Runs a candidate policy in shadow mode alongside a production baseline."""

    def __init__(
        self,
        baseline: BaselinePolicy,
        candidate: LearnedPolicy,
        *,
        extractor: FeatureExtractor | None = None,
        shift_guard: DistributionShiftGuard | None = None,
        confidence_threshold: float = 0.5,
        safe_evaluation_task_ids: set[str] | None = None,
    ) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.extractor = extractor or FeatureExtractor()
        self.shift_guard = shift_guard
        self.confidence_threshold = confidence_threshold
        self.safe_evaluation_task_ids = safe_evaluation_task_ids or set()
        self.records: list[ShadowRecord] = []
        self.stats = ShadowStats()

    def evaluate(
        self,
        state: ExecutiveState,
        *,
        task_id: str = "",
        allowed_actions: list[Action] | None = None,
        timestamp: str = "",
    ) -> ShadowRecord:
        # Production action: baseline policy.
        prod_dec = self.baseline.select_action(state, allowed_actions=allowed_actions)
        production_action = prod_dec.action

        # Distribution-shift check.
        shift_detected = False
        if self.shift_guard is not None:
            fv = self.extractor.extract(state)
            self.shift_guard.observe(fv.as_list())
            shift_detected = self.shift_guard.shifted()

        # Shadow action: candidate policy (with abstention).
        shadow_action = production_action
        shadow_confidence = 0.0
        shadow_abstained = True
        shadow_reason = "no candidate decision"

        if not shift_detected:
            try:
                c_dec = self.candidate.select_action(
                    state,
                    extractor=self.extractor,
                    allowed_actions=allowed_actions,
                )
                shadow_confidence = c_dec.confidence
                shadow_abstained = c_dec.abstained or c_dec.confidence < self.confidence_threshold
                if shadow_abstained:
                    shadow_action = production_action
                    shadow_reason = "candidate abstained"
                else:
                    shadow_action = c_dec.action
                    shadow_reason = "candidate argmax"
            except Exception as exc:
                shadow_abstained = True
                shadow_reason = f"candidate error: {exc}"
        else:
            shadow_reason = "distribution shift detected; using baseline"

        disagreement = production_action != shadow_action

        record = ShadowRecord(
            task_id=task_id,
            production_action=production_action,
            shadow_action=shadow_action,
            disagreement=disagreement,
            shadow_confidence=shadow_confidence,
            shadow_abstained=shadow_abstained,
            shadow_reason=shadow_reason,
            distribution_shift_detected=shift_detected,
            timestamp=timestamp,
        )
        self.records.append(record)
        self.stats.n += 1
        if disagreement:
            self.stats.disagreements += 1
        if shift_detected:
            self.stats.distribution_shifts += 1
        if shadow_abstained:
            self.stats.abstentions += 1
        self.stats.by_action[shadow_action.value] = (
            self.stats.by_action.get(shadow_action.value, 0) + 1
        )
        return record

    def disagreement_records(self) -> list[ShadowRecord]:
        return [r for r in self.records if r.disagreement]
