"""Build the real qualification runtime (v0.3.1 Section 6).

Constructs the actual Capsule Brain application using the same production
service bootstrap path used by normal execution.
"""
from __future__ import annotations

from typing import Any

from capsule_brain.autolearn.counterfactual import (
    CounterfactualIsolationFactory,
    RealCounterfactualRuntime,
)
from qualification.autolearn_v031.config import QualificationConfig


def build_real_qualification_runtime(
    config: QualificationConfig,
    *,
    conversation_service: Any = None,
    service_factory: Any = None,
) -> RealCounterfactualRuntime:
    """Build a RealCounterfactualRuntime using real Capsule Brain services.

    This must construct the actual application using the same production
    service bootstrap path. When a service_factory is provided, each
    execution creates a fresh, isolated service.
    """
    if conversation_service is None and service_factory is None:
        raise ValueError(
            "build_real_qualification_runtime requires either "
            "conversation_service or service_factory"
        )
    return RealCounterfactualRuntime(
        conversation_service=conversation_service,
        service_factory=service_factory,
    )


def build_isolation_factory(base_dir: str = "") -> CounterfactualIsolationFactory:
    """Build a CounterfactualIsolationFactory for isolated executions."""
    return CounterfactualIsolationFactory(base_dir=base_dir)


def assert_real_runtime(runtime: RealCounterfactualRuntime) -> None:
    """Assert that the runtime is real, not simulated."""
    if runtime.runtime_type != "real":
        raise RuntimeError(
            f"Official qualification requires runtime_type='real', "
            f"got '{runtime.runtime_type}'."
        )
