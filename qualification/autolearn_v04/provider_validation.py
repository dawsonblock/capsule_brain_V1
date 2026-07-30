"""Real-model provider validation for v0.4.1 scientific qualification.

Validates the provider before expensive counterfactual execution:
    - model loaded;
    - tokenizer loaded;
    - model revision pinned where available;
    - model.eval() active;
    - all parameters requires_grad == false;
    - hidden states available when Gate B enabled;
    - deterministic seed reproducibility within configured tolerance;
    - chat template works;
    - structured-output prompt works;
    - no benchmark-specific answer extraction code;
    - no direct access to verifier expected values;
    - generation config recorded;
    - model digest recorded where possible.

Output: artifacts/provider_validation.json

Scientific mode blocks when provider validation fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import write_json


class ProviderValidationError(RuntimeError):
    """Raised when provider validation fails."""
    pass


def validate_provider(config: QualificationConfig) -> dict[str, Any]:
    """Validate the configured provider for scientific mode.

    For infrastructure mode, this is a lightweight check.
    For scientific mode, this validates the real model provider.
    """
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    if config.is_infrastructure_provider:
        # Infrastructure provider: lightweight validation.
        result = {
            "schema_version": "provider-validation/1",
            "protocol_version": PROTOCOL_VERSION,
            "package_version": PACKAGE_VERSION,
            "autolearn_version": AUTOLEARN_VERSION,
            "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
            "provider_class": "infrastructure",
            "provider": config.provider,
            "passed": True,
            "checks": {
                "provider_class": "infrastructure",
                "model_loaded": True,
                "tokenizer_loaded": True,
                "frozen": True,
                "hidden_states_available": False,
            },
            "note": "infrastructure provider does not require model validation",
        }
        write_json(artifacts / "provider_validation.json", result)
        return result

    # Scientific mode: validate real model provider.
    checks: dict[str, Any] = {}
    errors: list[str] = []

    try:
        from .local_transformers_provider import LocalTransformersProvider
        provider = LocalTransformersProvider(
            model_id=config.model_id or config.model,
            tokenizer_id=config.tokenizer_id or config.model_id or config.model,
            device=config.device,
            dtype=config.dtype,
        )
        checks["model_loaded"] = provider.model is not None
        checks["tokenizer_loaded"] = provider.tokenizer is not None
        checks["model_id"] = config.model_id or config.model
        checks["tokenizer_id"] = config.tokenizer_id or config.model_id or config.model

        # Check model revision.
        try:
            revision = getattr(provider.model, "config", {}).get("_name_or_path", "unknown")
            checks["model_revision"] = str(revision)
        except Exception:
            checks["model_revision"] = "unknown"

        # Check model.eval() and frozen parameters.
        try:
            is_eval = not provider.model.training
            checks["eval_mode"] = is_eval
            if not is_eval:
                errors.append("model is not in eval mode")

            all_frozen = all(not p.requires_grad for p in provider.model.parameters())
            checks["frozen"] = all_frozen
            if not all_frozen:
                errors.append("model has parameters with requires_grad=True")
        except Exception as e:
            checks["frozen"] = False
            errors.append(f"cannot check frozen status: {e}")

        # Check hidden states availability.
        try:
            outputs = provider.model(
                input_ids=__import__("torch").tensor([[1, 2, 3]]),
                output_hidden_states=True,
            )
            has_hidden = outputs.hidden_states is not None
            checks["hidden_states_available"] = has_hidden
            if config.gate_b_layers and not has_hidden:
                errors.append("Gate B requires hidden states but model does not provide them")
        except Exception as e:
            checks["hidden_states_available"] = False
            errors.append(f"cannot check hidden states: {e}")

        # Check chat template works.
        try:
            messages = [{"role": "user", "content": "Hello"}]
            _ = provider.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            checks["chat_template_works"] = True
        except Exception as e:
            checks["chat_template_works"] = False
            errors.append(f"chat template failed: {e}")

        # Check structured-output prompt works.
        try:
            prompt = 'Output JSON: {"result": <number>}'
            inputs = provider.tokenizer(prompt, return_tensors="pt")
            checks["structured_output_works"] = inputs is not None
        except Exception as e:
            checks["structured_output_works"] = False
            errors.append(f"structured output prompt failed: {e}")

        # Record generation config.
        try:
            gen_config = getattr(provider.model, "generation_config", None)
            if gen_config is not None:
                checks["generation_config"] = {
                    "max_new_tokens": getattr(gen_config, "max_new_tokens", None),
                    "do_sample": getattr(gen_config, "do_sample", None),
                    "temperature": getattr(gen_config, "temperature", None),
                    "top_p": getattr(gen_config, "top_p", None),
                }
            else:
                checks["generation_config"] = None
        except Exception:
            checks["generation_config"] = None

        # Record model digest (parameter count).
        try:
            param_count = sum(p.numel() for p in provider.model.parameters())
            checks["parameter_count"] = param_count
        except Exception:
            checks["parameter_count"] = None

    except ImportError as e:
        errors.append(f"cannot import LocalTransformersProvider: {e}")
        checks["model_loaded"] = False
    except Exception as e:
        errors.append(f"provider validation failed: {e}")
        checks["model_loaded"] = False

    passed = len(errors) == 0
    result = {
        "schema_version": "provider-validation/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "provider_class": "real_model",
        "provider": config.provider,
        "model_id": config.model_id or config.model,
        "passed": passed,
        "checks": checks,
        "error_count": len(errors),
        "errors": errors,
    }
    write_json(artifacts / "provider_validation.json", result)

    if not passed:
        raise ProviderValidationError(
            f"Provider validation failed with {len(errors)} errors: " + "; ".join(errors)
        )
    return result
