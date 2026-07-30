"""Build the real qualification runtime for AutoLearn v0.4.0 (Section 5/11).

Constructs the actual Capsule Brain application using the normal bootstrap
path (build_application) to prove the runtime is real, then wraps a
RealQualificationRuntime whose service_factory is the
QualificationServiceFactory for per-action isolation.

Section 11.2: the deterministic grounded provider is the RUNTIME
INTEGRATION PROVIDER. It may only qualify infrastructure. A real model
backend (LocalTransformersProvider) is required for Gate A.
"""
from __future__ import annotations

from typing import Any

from capsule_brain.autolearn.counterfactual import (
    RealCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.runtime.bootstrap import build_application

from .config import (
    PROVIDER_INFRASTRUCTURE,
    PROVIDER_LOCAL_TRANSFORMERS,
    QualificationConfig,
)
from .grounded_provider import GroundedQualificationProvider
from .provisioners import QualificationServiceFactory


class RealQualificationRuntime:
    """Wraps the real runtime + per-action isolated service factory.

    ``runtime_type`` is always "real". The provider_type records whether
    the LLM backend is the infrastructure-only deterministic provider or a
    real model backend.
    """

    runtime_type: str = "real"

    def __init__(
        self,
        config: QualificationConfig,
        *,
        factory: QualificationServiceFactory,
        provider_type: str,
    ) -> None:
        self.config = config
        self.factory = factory
        self.provider_type = provider_type
        self._wrapped = RealCounterfactualRuntime(service_factory=_FactoryAdapter(factory))

    @property
    def is_infrastructure_provider(self) -> bool:
        return self.provider_type in (PROVIDER_INFRASTRUCTURE, "qual_grounded")

    @property
    def is_real_model_provider(self) -> bool:
        return self.provider_type == PROVIDER_LOCAL_TRANSFORMERS

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


def _make_llm_providers(config: QualificationConfig) -> tuple[dict[str, Any], str]:
    """Build the LLM provider map and return (providers, provider_type).

    For the infrastructure provider, returns the deterministic grounded
    provider. For local_transformers, attempts to construct the real model
    provider; if transformers is not installed, raises with a clear message.
    """
    if config.provider == PROVIDER_LOCAL_TRANSFORMERS:
        from .local_transformers_provider import LocalTransformersProvider
        provider = LocalTransformersProvider(
            model_id=config.model,
            device=config.device,
            dtype=config.dtype,
        )
        return {provider.name: provider}, PROVIDER_LOCAL_TRANSFORMERS
    # Default: infrastructure integration provider.
    provider = GroundedQualificationProvider()
    return {provider.name: provider}, PROVIDER_INFRASTRUCTURE


def build_real_qualification_runtime(config: QualificationConfig) -> RealQualificationRuntime:
    """Build a real qualification runtime using normal bootstrap paths."""
    llm_providers, provider_type = _make_llm_providers(config)
    # Prove the normal bootstrap path works.
    app = build_application(
        config.to_capsule_config(),
        llm_providers=llm_providers,
    )
    factory = QualificationServiceFactory(config, sandbox_root=config.sandbox_root)
    runtime = RealQualificationRuntime(config, factory=factory, provider_type=provider_type)
    assert_runtime_is_real(runtime)
    return runtime


def assert_real_runtime(runtime: Any) -> None:
    rt_type = getattr(runtime, "runtime_type", "unknown")
    if rt_type != "real":
        raise RuntimeError(
            f"Official qualification requires runtime_type='real', got '{rt_type}'."
        )
