"""Deterministic, interpretable state feature extractor.

v0.1 does not train on transformer hidden states. It uses interpretable,
deterministic features so the same state always produces the same feature
vector. The schema is versioned; an incompatible policy is rejected at load
time (see policy.py / service.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from capsule_brain.autolearn.schema import ExecutiveState

# Bump when the set/ordering/semantics of features change. Policies record
# the schema version they were trained against and are rejected at load time
# if it does not match.
FEATURE_SCHEMA_VERSION = "exec_features_v1"

# Ordered list of feature names — the canonical column order. The vector
# returned by FeatureExtractor.extract() always follows this order.
FEATURE_NAMES: tuple[str, ...] = (
    "prompt_length",
    "log_prompt_length",
    "code_indicators",
    "arithmetic_indicators",
    "structured_output_request",
    "tool_required_keywords",
    "retrieval_indicators",
    "prior_conversation_depth",
    "semantic_memory_hit_count",
    "semantic_memory_similarity",
    "num_available_tools",
    "workflow_available",
    "previous_attempt_failed",
    "verification_failure_type_id",
    "model_identity_id",
    "estimated_difficulty",
    "workflow_capability_match",
)

# Keyword sets. Lowercased, matched as substrings.
_TOOL_KEYWORDS = (
    "tool", "call", "fetch", "lookup", "query", "api", "search",
    "retrieve", "compute", "calculate", "execute", "run", "secret",
    "nonce", "current", "latest", "live", "price", "weather", "stock",
)
_RETRIEVAL_KEYWORDS = (
    "remember", "recall", "earlier", "previously", "last time",
    "you said", "we discussed", "from memory", "stored", "persisted",
)
_CODE_KEYWORDS = (
    "def ", "class ", "import ", "function ", "pytest", "unittest",
    "```", "return ", "lambda", "async ", "await ", "print(",
    "fix the bug", "implement", "refactor", "write a test", "code",
)
_STRUCTURED_KEYWORDS = (
    "json", "yaml", "xml", "csv", "table", "schema", "structured",
    "list of", "as a dict", "as a list",
)
_ARITHMETIC_REGEX = re.compile(r"\b\d+(?:\.\d+)?\s*[+\-*/^%]|\b\d+(?:\.\d+)?\b")


def _count_matches(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def _bool_to_float(b: bool) -> float:
    return 1.0 if b else 0.0


@dataclass(slots=True)
class FeatureVector:
    """A versioned feature vector with named columns."""

    schema_version: str
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_list(self) -> list[float]:
        return list(self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values))


class FeatureExtractor:
    """Deterministic feature extractor.

    The same ExecutiveState always yields the same FeatureVector. No
    randomness, no external calls, no model hidden states.
    """

    schema_version: str = FEATURE_SCHEMA_VERSION
    names: tuple[str, ...] = FEATURE_NAMES

    # Stable categorical encodings for non-boolean fields.
    _VERIFICATION_FAILURE_TYPES: tuple[str, ...] = (
        "none", "syntax", "acceptance", "consistency", "required_fields",
        "execution", "timeout", "other",
    )
    _MODEL_IDENTITIES: tuple[str, ...] = (
        "unknown", "qwen2.5-coder:3b", "llama3.1:8b", "gpt-4o-mini",
        "fake", "other",
    )

    def extract(self, state: ExecutiveState) -> FeatureVector:
        pf = state.prompt_features or {}
        cf = state.conversation_features or {}
        mf = state.memory_features or {}

        prompt_text = str(pf.get("text", "") or "")
        prompt_length = float(len(prompt_text))
        log_prompt_length = float(_safe_log(prompt_length))

        code_indicators = float(_count_matches(prompt_text, _CODE_KEYWORDS))
        arithmetic_indicators = float(
            len(_ARITHMETIC_REGEX.findall(prompt_text))
        )
        structured_output_request = _bool_to_float(
            bool(_count_matches(prompt_text, _STRUCTURED_KEYWORDS))
        )
        tool_required_keywords = float(
            _count_matches(prompt_text, _TOOL_KEYWORDS)
        )
        retrieval_indicators = float(
            _count_matches(prompt_text, _RETRIEVAL_KEYWORDS)
        )

        prior_conversation_depth = float(
            int(cf.get("depth", 0) or 0)
        )
        semantic_memory_hit_count = float(
            int(mf.get("hit_count", 0) or 0)
        )
        semantic_memory_similarity = float(
            _clip(mf.get("top_similarity", 0.0) or 0.0, 0.0, 1.0)
        )

        num_available_tools = float(len(state.available_tools))
        workflow_available = _bool_to_float(state.workflow_available)

        previous_attempt_failed = _bool_to_float(
            bool(pf.get("previous_attempt_failed", False))
        )
        verification_failure_type_id = float(
            self._encode_category(
                str(pf.get("verification_failure_type", "none") or "none"),
                self._VERIFICATION_FAILURE_TYPES,
            )
        )
        model_identity_id = float(
            self._encode_category(
                str(state.model_id or "unknown"),
                self._MODEL_IDENTITIES,
            )
        )
        estimated_difficulty = float(
            _clip(pf.get("estimated_difficulty", 0.5) or 0.5, 0.0, 1.0)
        )
        workflow_capability_match = _bool_to_float(
            bool(pf.get("workflow_capability_match", False))
        )

        values = (
            prompt_length,
            log_prompt_length,
            code_indicators,
            arithmetic_indicators,
            structured_output_request,
            tool_required_keywords,
            retrieval_indicators,
            prior_conversation_depth,
            semantic_memory_hit_count,
            semantic_memory_similarity,
            num_available_tools,
            workflow_available,
            previous_attempt_failed,
            verification_failure_type_id,
            model_identity_id,
            estimated_difficulty,
            workflow_capability_match,
        )
        return FeatureVector(
            schema_version=self.schema_version,
            names=self.names,
            values=values,
        )

    @staticmethod
    def _encode_category(value: str, categories: tuple[str, ...]) -> int:
        lowered = value.lower()
        if lowered in categories:
            return categories.index(lowered)
        # "other" is conventionally the last category; fall back to it if
        # present, otherwise 0.
        if "other" in categories:
            return categories.index("other")
        return 0


def _safe_log(x: float) -> float:
    if x <= 0:
        return 0.0
    import math

    return math.log1p(x)


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
