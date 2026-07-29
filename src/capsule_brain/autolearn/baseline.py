"""Deterministic rule-based baseline executive policy.

This is the concrete policy that AutoLearn must measurably beat on held-out,
independently verified tasks. Every baseline decision is recorded in the
ExperienceStore so the learned router is always compared against the same
reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from capsule_brain.autolearn.features import FeatureExtractor, FeatureVector
from capsule_brain.autolearn.schema import Action, ExecutiveState

BASELINE_POLICY_VERSION = "baseline_v1"


@dataclass(slots=True)
class BaselineDecision:
    action: Action
    scores: dict[str, float]
    reason: str
    policy_version: str = BASELINE_POLICY_VERSION
    feature_vector: FeatureVector | None = None


class BaselinePolicy:
    """Rule-based baseline router.

    The rules intentionally mirror the routing intuition a careful human
    operator would use, expressed as deterministic thresholds over the
    interpretable feature vector. They are deliberately *not* tuned to be
    optimal — they are the reference that the learned policy must beat.
    """

    version: str = BASELINE_POLICY_VERSION
    name: str = "baseline"

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        self.extractor = extractor or FeatureExtractor()

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> BaselineDecision:
        fv = self.extractor.extract(state)
        d = fv.as_dict()

        allowed = set(allowed_actions) if allowed_actions is not None else set(Action.all())

        # Build a score for every allowed action so the decision is
        # auditable even when the rule picks a different action.
        scores: dict[str, float] = {a.value: 0.0 for a in Action.all()}

        # Strong tool signal: explicit tool keywords, secret/nonce/live data,
        # or arithmetic that benefits from external computation.
        tool_signal = (
            d["tool_required_keywords"]
            + 0.5 * d["arithmetic_indicators"]
            + 0.25 * d["structured_output_request"]
        )
        scores[Action.CALL_TOOL.value] = tool_signal

        # Memory retrieval signal: retrieval keywords or strong semantic hit.
        memory_signal = (
            d["retrieval_indicators"]
            + 2.0 * d["semantic_memory_similarity"]
            + 0.5 * d["semantic_memory_hit_count"]
        )
        scores[Action.RETRIEVE_MEMORY.value] = memory_signal

        # Workflow signal: code indicators + workflow available + capability match.
        workflow_signal = (
            d["code_indicators"]
            + 0.5 * d["workflow_capability_match"]
        ) * d["workflow_available"]
        scores[Action.START_WORKFLOW.value] = workflow_signal

        # Reflect signal: previous attempt failed verification.
        reflect_signal = (
            2.0 * d["previous_attempt_failed"]
            + 0.5 * d["verification_failure_type_id"]
        )
        scores[Action.REFLECT.value] = reflect_signal

        # Direct answer is the default; penalize it when stronger signals exist.
        direct_signal = 1.0 - 0.3 * (
            tool_signal + memory_signal + workflow_signal + reflect_signal
        )
        scores[Action.ANSWER_DIRECT.value] = max(0.0, direct_signal)

        # Operator escalation is reserved for safety/ambiguous cases.
        scores[Action.ASK_OPERATOR.value] = 0.0

        # Apply the rule precedence from the spec.
        chosen: Action
        reason: str

        if Action.REFLECT in allowed and d["previous_attempt_failed"] >= 1.0:
            chosen = Action.REFLECT
            reason = "previous attempt failed verification"
        elif Action.CALL_TOOL in allowed and tool_signal >= 1.5:
            chosen = Action.CALL_TOOL
            reason = "task requires exact external data (tool keywords/secret)"
        elif (
            Action.START_WORKFLOW in allowed
            and d["workflow_available"] >= 1.0
            and workflow_signal >= 1.5
        ):
            chosen = Action.START_WORKFLOW
            reason = "task matches coding workflow"
        elif Action.RETRIEVE_MEMORY in allowed and memory_signal >= 1.5:
            chosen = Action.RETRIEVE_MEMORY
            reason = "memory similarity above threshold"
        elif Action.ANSWER_DIRECT in allowed:
            chosen = Action.ANSWER_DIRECT
            reason = "no stronger signal; default direct answer"
        else:
            # Fall back to the highest-scoring allowed action.
            allowed_sorted = sorted(
                ((a, scores.get(a.value, 0.0)) for a in allowed),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if allowed_sorted:
                chosen = allowed_sorted[0][0]
                reason = "fallback to highest-scoring allowed action"
            else:
                chosen = Action.ASK_OPERATOR
                reason = "no allowed actions; escalate to operator"

        return BaselineDecision(
            action=chosen,
            scores=scores,
            reason=reason,
            policy_version=self.version,
            feature_vector=fv,
        )
