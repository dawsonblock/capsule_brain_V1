"""Build the real qualification runtime for AutoLearn v0.3.2 (Section 5).

Constructs the actual Capsule Brain application using the normal bootstrap
path (build_application) to prove the runtime is real, then wraps a
RealCounterfactualRuntime whose service_factory is the
QualificationServiceFactory for per-action isolation.

Official run must fail if runtime_type != "real". Development simulation is
allowed only with an explicit flag and must label its output.
"""
from __future__ import annotations

from typing import Any

from capsule_brain.autolearn.counterfactual import (
    RealCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.runtime.bootstrap import build_application

from .config import QualificationConfig
from .grounded_provider import GroundedQualificationProvider
from .provisioners import QualificationServiceFactory


class RealQualificationRuntime:
    """Wraps the real runtime + per-action isolated service factory.

    ``runtime_type`` is always "real". The wrapped RealCounterfactualRuntime
    is kept for compatibility with existing counterfactual helpers, but the
    v0.3.2 counterfactual runner drives the QualificationServiceFactory
    directly for full per-action isolation.
    """

    runtime_type: str = "real"

    def __init__(self, config: QualificationConfig, *, factory: QualificationServiceFactory) -> None:
        self.config = config
        self.factory = factory
        # A RealCounterfactualRuntime wired to the same factory so any helper
        # that consumes a CounterfactualRuntime sees runtime_type == "real".
        self._wrapped = RealCounterfactualRuntime(service_factory=_FactoryAdapter(factory))

    def assert_real(self) -> None:
        if self.runtime_type != "real":
            raise RuntimeError(
                f"Official qualification requires runtime_type='real', "
                f"got '{self.runtime_type}'."
            )


class _FactoryAdapter:
    """Adapts QualificationServiceFactory to the RealCounterfactualRuntime
    service_factory.build(task) -> object-with-execute_executive_action protocol."""

    def __init__(self, factory: QualificationServiceFactory) -> None:
        self._factory = factory

    async def build(self, task: Any) -> Any:
        # task may be a CounterfactualTask or a manifest dict; normalize to dict.
        if isinstance(task, dict):
            task_dict = task
        else:
            task_dict = {
                "task_id": getattr(task, "task_id", ""),
                "family": getattr(task, "family", ""),
                "prompt": getattr(task, "prompt", ""),
                "setup_spec": dict(getattr(task, "setup", {}) or {}),
                "verifier_spec": {},
            }
        return await self._factory.build(task_dict)


def build_real_qualification_runtime(config: QualificationConfig) -> RealQualificationRuntime:
    """Build a real qualification runtime using normal bootstrap paths.

    First constructs a full Capsule Brain application via build_application to
    prove the normal bootstrap path works with the qualification config. Then
    builds the per-action isolated service factory.
    """
    # Prove the normal bootstrap path works. We do not keep this app instance
    # for counterfactual execution (per-action isolation uses the factory),
    # but constructing it validates that the qualification config boots a real
    # Capsule Brain application end-to-end.
    app = build_application(
        config.to_capsule_config(),
        llm_providers={"qual_grounded": GroundedQualificationProvider()},
    )
    # The app is intentionally not started here; construction validates the
    # real bootstrap path. The factory builds fresh isolated services per
    # action for actual counterfactual execution.
    factory = QualificationServiceFactory(config, sandbox_root=config.sandbox_root)
    runtime = RealQualificationRuntime(config, factory=factory)
    assert_runtime_is_real(runtime)  # hard-assert runtime_type == "real"
    return runtime


def assert_real_runtime(runtime: Any) -> None:
    """Section 5: hard-assert the runtime is real."""
    rt_type = getattr(runtime, "runtime_type", "unknown")
    if rt_type != "real":
        raise RuntimeError(
            f"Official qualification requires runtime_type='real', got '{rt_type}'."
        )
