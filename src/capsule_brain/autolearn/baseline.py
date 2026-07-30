"""Deterministic rule-based baseline executive policy.

This is the concrete policy that AutoLearn must measurably beat on held-out,
independently verified tasks. Every baseline decision is recorded in the
ExperienceStore so the learned router is always compared against the same
reference.

v0.2 ships ``BaselinePolicyV2`` — a competent deterministic router with
documented rules for tool-required tasks, memory retrieval, workflow/code
tasks, and obvious safety/operator escalation. The learned policy must beat
a competent baseline, not a deliberately crippled one. ``BaselinePolicy``
(v1) is retained for backward compatibility with existing v0.1 artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from capsule_brain.autolearn.features import FeatureExtractor, FeatureVector
from capsule_brain.autolearn.schema import Action, ExecutiveState

BASELINE_POLICY_VERSION = "baseline_v1"
BASELINE_V2_POLICY_VERSION = "baseline_v2"
BASELINE_V3_POLICY_VERSION = "baseline_v3"


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


# ---------------------------------------------------------------------------
# BaselinePolicyV2 — competent deterministic router (v0.2)
# ---------------------------------------------------------------------------


# Safety/operator keywords. A prompt matching any of these is escalated to
# ASK_OPERATOR regardless of other signals. This is the safety floor; the
# learned policy never controls it.
_SAFETY_KEYWORDS = (
    "delete the production", "drop the", "rm -rf", "force-push",
    "disable the firewall", "exfiltrate", "backdoor", "reverse shell",
    "cryptominer", "phishing", "harvest", "bypass the verification",
    "disable all safety", "disable container", "rewrite the policy",
    "mark this experience successful without", "scrape all user passwords",
    "ssh keys", "browser cookies", "cryptocurrency wallet",
    "unsafe_execution", "bypass the safety", "overwrite the main branch",
)


class BaselinePolicyV2(BaselinePolicy):
    """Competent deterministic router with documented rules.

    The learned policy must beat this, not a crippled baseline. Rules
    (evaluated in order, first match wins):

    1. SAFETY: prompt matches a safety keyword -> ASK_OPERATOR.
       (Hard safety floor; never learned.)
    2. REFLECTION: previous attempt failed verification -> REFLECT.
    3. TOOL: strong tool signal (secret/nonce/live/current keywords + a
       tool is available) -> CALL_TOOL.
    4. WORKFLOW: code indicators + workflow available + capability match
       -> START_WORKFLOW.
    5. MEMORY: retrieval keywords or strong semantic memory hit
       -> RETRIEVE_MEMORY.
    6. DEFAULT: ANSWER_DIRECT.

    Each rule is documented and frozen before evaluation. The threshold
    values are deliberately interpretable.
    """

    version: str = BASELINE_V2_POLICY_VERSION
    name: str = "baseline_v2"

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> BaselineDecision:
        fv = self.extractor.extract(state)
        d = fv.as_dict()
        allowed = set(allowed_actions) if allowed_actions is not None else set(Action.all())
        prompt_text = str(state.prompt_features.get("text", "") or "").lower()

        scores: dict[str, float] = {a.value: 0.0 for a in Action.all()}

        # Rule 1: SAFETY — hard floor. Checked first and never overridden.
        safety_match = any(kw in prompt_text for kw in _SAFETY_KEYWORDS)
        if safety_match:
            scores[Action.ASK_OPERATOR.value] = 10.0
            chosen = Action.ASK_OPERATOR if Action.ASK_OPERATOR in allowed else self._best_allowed(scores, allowed)
            return BaselineDecision(
                action=chosen,
                scores=scores,
                reason="safety keyword match; operator escalation",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 2: REFLECTION — previous attempt failed verification.
        if d["previous_attempt_failed"] >= 1.0 and Action.REFLECT in allowed:
            scores[Action.REFLECT.value] = 5.0
            return BaselineDecision(
                action=Action.REFLECT,
                scores=scores,
                reason="previous attempt failed verification; reflect",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 3: TOOL — strong tool signal. Requires a tool to be available
        # AND a strong keyword hit (secret/nonce/live/current/latest/fetch).
        tool_signal = (
            d["tool_required_keywords"]
            + 0.5 * d["arithmetic_indicators"]
            + 0.25 * d["structured_output_request"]
        )
        scores[Action.CALL_TOOL.value] = tool_signal
        has_tool = d["num_available_tools"] >= 1.0
        if Action.CALL_TOOL in allowed and has_tool and tool_signal >= 1.5:
            return BaselineDecision(
                action=Action.CALL_TOOL,
                scores=scores,
                reason="tool signal above threshold with tool available",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 4: WORKFLOW — code indicators + workflow available + match.
        workflow_signal = (
            d["code_indicators"]
            + 0.5 * d["workflow_capability_match"]
        ) * d["workflow_available"]
        scores[Action.START_WORKFLOW.value] = workflow_signal
        if (
            Action.START_WORKFLOW in allowed
            and d["workflow_available"] >= 1.0
            and workflow_signal >= 1.5
        ):
            return BaselineDecision(
                action=Action.START_WORKFLOW,
                scores=scores,
                reason="code indicators with workflow available and capability match",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 5: MEMORY — retrieval keywords or strong semantic hit.
        memory_signal = (
            d["retrieval_indicators"]
            + 2.0 * d["semantic_memory_similarity"]
            + 0.5 * d["semantic_memory_hit_count"]
        )
        scores[Action.RETRIEVE_MEMORY.value] = memory_signal
        if Action.RETRIEVE_MEMORY in allowed and memory_signal >= 1.5:
            return BaselineDecision(
                action=Action.RETRIEVE_MEMORY,
                scores=scores,
                reason="memory retrieval signal above threshold",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 6: DEFAULT — direct answer.
        scores[Action.ANSWER_DIRECT.value] = max(0.0, 1.0 - 0.3 * (
            tool_signal + memory_signal + workflow_signal
        ))
        if Action.ANSWER_DIRECT in allowed:
            return BaselineDecision(
                action=Action.ANSWER_DIRECT,
                scores=scores,
                reason="no stronger signal; default direct answer",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Fallback: highest-scoring allowed action.
        chosen = self._best_allowed(scores, allowed)
        return BaselineDecision(
            action=chosen,
            scores=scores,
            reason="fallback to highest-scoring allowed action",
            policy_version=self.version,
            feature_vector=fv,
        )

    @staticmethod
    def _best_allowed(scores: dict[str, float], allowed: set[Action]) -> Action:
        allowed_sorted = sorted(
            ((a, scores.get(a.value, 0.0)) for a in allowed),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if allowed_sorted:
            return allowed_sorted[0][0]
        return Action.ASK_OPERATOR


# ---------------------------------------------------------------------------
# v0.3.1 Section 15: Frozen BaselinePolicyV3
# ---------------------------------------------------------------------------

# Additional safety keywords for v3 (broader coverage).
_SAFETY_KEYWORDS_V3 = _SAFETY_KEYWORDS + (
    "delete production database", "drop the database", "wipe all data",
    "disable all safety", "bypass all verification", "remove all audit",
    "exfiltrate all data", "steal all credentials", "install malware",
    "create backdoor", "escalate privileges", "overwrite active policy",
)


class BaselinePolicyV3(BaselinePolicyV2):
    """v0.3.1 Section 15: Frozen competent baseline.

    A frozen, versioned baseline that must be created and stored BEFORE
    generating candidate results. It must not be edited after seeing
    final-test outcomes.

    v3 changes from v2:
    - Broader safety keyword coverage (includes SAFE_REFUSAL for outright
      security bypasses and exfiltration).
    - Slightly tuned thresholds for workflow and memory signals.
    - Versioned source digest for provenance.
    - SAFE_REFUSAL is used for outright refusals (security bypass,
      exfiltration) instead of always escalating to ASK_OPERATOR.
    """

    version: str = BASELINE_V3_POLICY_VERSION
    name: str = "baseline_v3"

    # v3: Refusal keywords that warrant SAFE_REFUSAL, not just ASK_OPERATOR.
    _REFUSAL_KEYWORDS: tuple[str, ...] = (
        "disable the firewall", "disable all safety", "disable container",
        "bypass the verification", "bypass the safety", "disable verification",
        "rewrite the policy", "overwrite the safety", "remove audit logs",
        "unsafe_execution", "exfiltrate", "backdoor", "reverse shell",
        "cryptominer", "phishing", "harvest", "scrape all user passwords",
        "ssh keys", "browser cookies", "cryptocurrency wallet",
        "send unauthorized email", "install persistence",
        "execute untrusted network command", "install malware",
        "create backdoor", "steal all credentials",
    )

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> BaselineDecision:
        fv = self.extractor.extract(state)
        d = fv.as_dict()
        allowed = set(allowed_actions) if allowed_actions is not None else set(Action.all())
        prompt_text = str(state.prompt_features.get("text", "") or "").lower()

        scores: dict[str, float] = {a.value: 0.0 for a in Action.all()}

        # Rule 1: SAFETY — hard floor. v3 distinguishes REFUSE from ESCALATE.
        refusal_match = any(kw in prompt_text for kw in self._REFUSAL_KEYWORDS)
        if refusal_match:
            if Action.SAFE_REFUSAL in allowed:
                scores[Action.SAFE_REFUSAL.value] = 10.0
                return BaselineDecision(
                    action=Action.SAFE_REFUSAL,
                    scores=scores,
                    reason="safety refusal: security bypass or exfiltration detected",
                    policy_version=self.version,
                    feature_vector=fv,
                )
            # Fallback to ASK_OPERATOR if SAFE_REFUSAL not allowed.
            scores[Action.ASK_OPERATOR.value] = 10.0
            chosen = Action.ASK_OPERATOR if Action.ASK_OPERATOR in allowed else self._best_allowed(scores, allowed)
            return BaselineDecision(
                action=chosen,
                scores=scores,
                reason="safety refusal (SAFE_REFUSAL not allowed); operator escalation",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Check destructive keywords for operator escalation.
        destructive_match = any(kw in prompt_text for kw in _SAFETY_KEYWORDS_V3)
        if destructive_match:
            scores[Action.ASK_OPERATOR.value] = 10.0
            chosen = Action.ASK_OPERATOR if Action.ASK_OPERATOR in allowed else self._best_allowed(scores, allowed)
            return BaselineDecision(
                action=chosen,
                scores=scores,
                reason="safety keyword match; operator escalation",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 2: REFLECTION — previous attempt failed verification.
        if d["previous_attempt_failed"] >= 1.0 and Action.REFLECT in allowed:
            scores[Action.REFLECT.value] = 5.0
            return BaselineDecision(
                action=Action.REFLECT,
                scores=scores,
                reason="previous attempt failed verification; reflect",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 3: TOOL — strong tool signal. v3: slightly lower threshold.
        tool_signal = (
            d["tool_required_keywords"]
            + 0.5 * d["arithmetic_indicators"]
            + 0.25 * d["structured_output_request"]
        )
        scores[Action.CALL_TOOL.value] = tool_signal
        has_tool = d["num_available_tools"] >= 1.0
        if Action.CALL_TOOL in allowed and has_tool and tool_signal >= 1.3:
            return BaselineDecision(
                action=Action.CALL_TOOL,
                scores=scores,
                reason="tool signal above threshold with tool available",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 4: WORKFLOW — code indicators + workflow available + match.
        workflow_signal = (
            d["code_indicators"]
            + 0.5 * d["workflow_capability_match"]
        ) * d["workflow_available"]
        scores[Action.START_WORKFLOW.value] = workflow_signal
        if (
            Action.START_WORKFLOW in allowed
            and d["workflow_available"] >= 1.0
            and workflow_signal >= 1.3
        ):
            return BaselineDecision(
                action=Action.START_WORKFLOW,
                scores=scores,
                reason="code indicators with workflow available and capability match",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 5: MEMORY — retrieval keywords or strong semantic hit.
        memory_signal = (
            d["retrieval_indicators"]
            + 2.0 * d["semantic_memory_similarity"]
            + 0.5 * d["semantic_memory_hit_count"]
        )
        scores[Action.RETRIEVE_MEMORY.value] = memory_signal
        if Action.RETRIEVE_MEMORY in allowed and memory_signal >= 1.3:
            return BaselineDecision(
                action=Action.RETRIEVE_MEMORY,
                scores=scores,
                reason="memory retrieval signal above threshold",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Rule 6: DEFAULT — direct answer.
        scores[Action.ANSWER_DIRECT.value] = max(0.0, 1.0 - 0.3 * (
            tool_signal + memory_signal + workflow_signal
        ))
        if Action.ANSWER_DIRECT in allowed:
            return BaselineDecision(
                action=Action.ANSWER_DIRECT,
                scores=scores,
                reason="no stronger signal; default direct answer",
                policy_version=self.version,
                feature_vector=fv,
            )

        # Fallback: highest-scoring allowed action.
        chosen = self._best_allowed(scores, allowed)
        return BaselineDecision(
            action=chosen,
            scores=scores,
            reason="fallback to highest-scoring allowed action",
            policy_version=self.version,
            feature_vector=fv,
        )
