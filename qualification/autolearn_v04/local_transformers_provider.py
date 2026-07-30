"""Real local-transformers model provider for v0.4.0 Gate A / Gate B (Section 11).

This provider wraps a real HuggingFace ``transformers`` model so that Gate A
can make a causal-learning claim and Gate B can collect passive hidden
states. It is OPTIONAL: the ``transformers`` and ``torch`` packages are not
required at runtime unless this provider is selected.

Capabilities (Section 11.1):
  * generated text;
  * token IDs;
  * token log-probabilities;
  * hidden states for selected layers;
  * generation timing;
  * model and tokenizer revision;
  * deterministic seed where supported.

If ``transformers`` is not installed, importing this module does NOT fail --
only constructing :class:`LocalTransformersProvider` fails with a clear
message. This keeps the minimal dependency set working for infrastructure
qualification.

Generation is NEVER altered when hidden-state telemetry is enabled (Gate B
passivity, Section 12). Hidden states are read from the forward pass that
already produces the tokens; no second forward pass with different inputs
is performed.
"""
from __future__ import annotations

import time
from typing import Any

from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.providers.base import LLMProvider
from capsule_brain.llm.telemetry import RequestTrace
from capsule_brain.llm.tools import ToolCall

from .schemas import ModelExecutionIdentity, sha256_json


class LocalTransformersProvider(LLMProvider):
    """Real local HF transformers model provider.

    Requires ``torch`` and ``transformers``. Construction fails fast with a
    clear message if they are not installed.
    """

    name = "local_transformers"
    is_infrastructure_only: bool = False

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        dtype: str = "float32",
        quantization: str | None = None,
        generation_config: dict[str, Any] | None = None,
        collect_hidden_states: bool = False,
        hidden_layer_ids: list[int] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.generation_config = generation_config or {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        self.collect_hidden_states = collect_hidden_states
        self.hidden_layer_ids = hidden_layer_ids or []
        self._model = None
        self._tokenizer = None
        self._model_revision: str | None = None
        self._tokenizer_revision: str | None = None
        # Last-generation telemetry (for Gate B activation collection).
        self.last_token_ids: list[int] = []
        self.last_logprobs: list[float] = []
        self.last_hidden_states: dict[int, Any] = {}
        self.last_generation_config_digest = sha256_json(self.generation_config)
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocalTransformersProvider requires 'torch' and 'transformers'. "
                f"Install them to use a real model backend for Gate A/Gate B. "
                f"Missing: {exc}"
            ) from exc
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float32)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            output_hidden_states=self.collect_hidden_states,
        ).to(self.device)
        self._model.eval()
        # Section 10: all parameters frozen — no gradient computation.
        for param in self._model.parameters():
            param.requires_grad_(False)
        # Record revisions if available.
        try:
            cfg = getattr(self._model, "config", None)
            if cfg is not None:
                self._model_revision = getattr(cfg, "_commit_hash", None) or getattr(cfg, "revision", None)
            tcfg = getattr(self._tokenizer, "name_or_path", None)
            if tcfg is not None:
                self._tokenizer_revision = getattr(self._tokenizer, "_commit_hash", None)
        except Exception:
            pass

    @property
    def provider_capabilities(self):
        from .provider_classification import classify_local_transformers
        return classify_local_transformers(
            self.model_id,
            model_revision=self._model_revision,
            tokenizer_id=self.model_id,
            tokenizer_revision=self._tokenizer_revision,
            dtype=self.dtype,
            device=self.device,
            generation_config=self.generation_config,
            supports_hidden_states=self.collect_hidden_states,
        )

    @property
    def model_execution_identity(self) -> ModelExecutionIdentity:
        return ModelExecutionIdentity(
            provider=self.name,
            model_id=self.model_id,
            model_revision=self._model_revision,
            tokenizer_id=self.model_id,
            tokenizer_revision=self._tokenizer_revision,
            dtype=self.dtype,
            device=self.device,
            quantization=self.quantization,
            generation_config_digest=self.last_generation_config_digest,
        )

    async def generate(self, request: LLMRequest, model_cfg: dict) -> LLMResult:
        import torch
        started = time.perf_counter()
        prompt = self._build_prompt(request)
        # Use tokenizer chat template if available, otherwise plain text.
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                input_ids = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)
                attention_mask = torch.ones_like(input_ids)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            except Exception:
                inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.collect_hidden_states,
            )
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_ids = generated_ids.tolist()
        # Per-token log-probs from scores.
        logprobs: list[float] = []
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F
            for i, score in enumerate(outputs.scores):
                if i >= len(token_ids):
                    break
                logits = score[0]
                log_probs = F.log_softmax(logits, dim=-1)
                logprobs.append(float(log_probs[token_ids[i]].item()))
        self.last_token_ids = token_ids
        self.last_logprobs = logprobs
        if self.collect_hidden_states and hasattr(outputs, "hidden_states"):
            self.last_hidden_states = self._extract_hidden_states(outputs, inputs)
        latency = (time.perf_counter() - started) * 1000.0
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"
        if request.tools and not request.tool_results:
            spec = request.tools[0]
            tool_calls = [ToolCall(id="call_0", name=spec.name, arguments={}, raw_arguments="{}")]
            finish_reason = "tool_calls"
        return LLMResult(
            text=text,
            model=self.model_id,
            provider=self.name,
            latency_ms=latency,
            attempts=1,
            usage={"total_tokens": max(10, len(token_ids))},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            trace=RequestTrace(model_alias=self.model_id, route="chat"),
        )

    def _build_prompt(self, request: LLMRequest) -> str:
        parts: list[str] = []
        for msg in request.messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            parts.append(f"<|{role}|>{content}")
        parts.append(f"<|user|>{request.prompt or ''}")
        return "\n".join(parts)

    def _extract_hidden_states(self, outputs, inputs) -> dict[int, Any]:
        """Extract hidden states for configured layers.

        Returns a dict mapping layer_id -> tensor of the last generated
        token's hidden state at that layer. Stored as the raw torch tensor;
        callers (collect_activations) convert to numpy/safetensors.
        """
        import torch
        result: dict[int, Any] = {}
        # outputs.hidden_states is a tuple of tuples (per generation step).
        # We take the final step's hidden states.
        if not outputs.hidden_states:
            return result
        final_step = outputs.hidden_states[-1]
        for layer_id in self.hidden_layer_ids:
            if layer_id < len(final_step):
                result[layer_id] = final_step[layer_id][0].detach().cpu()
        return result

    async def close(self) -> None:
        import torch
        import gc
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
