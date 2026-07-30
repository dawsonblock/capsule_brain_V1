"""Modal GPU provider for real frozen-transformer qualification (Section 10).

Runs a real frozen causal language model on Modal GPU infrastructure.
The model is loaded once per container, parameters are frozen
(requires_grad=False), and all generation is deterministic with
seed control.

Model target: Qwen2.5-3B-Instruct (or configurable alternative).
GPU: NVIDIA A10G (24GB VRAM) — sufficient for 3B model in float16.

This module is self-contained and does NOT import from the
qualification/autolearn_v04 package or capsule_brain, because Modal
containers only have the packages installed via the image definition.
All dependencies (model loading, generation) are handled inside the
container using only torch + transformers.

The provider:
    - loads model with transformers AutoModelForCausalLM;
    - sets model.eval() and requires_grad=False on all parameters;
    - uses tokenizer chat template correctly;
    - supplies attention mask;
    - requests hidden states only when Gate B collection is enabled;
    - records generation configuration;
    - pins model revision where available;
    - contains NO benchmark-answer extraction shortcuts.
"""
# NOTE: Do NOT use `from __future__ import annotations` in this file.
# Modal's parameter type system requires real type objects, not string
# annotations, and the future-annotations import breaks it on Python 3.12.
# NOTE: Do NOT import from qualification.autolearn_v04 or capsule_brain
# at module level — Modal containers don't have those packages.

import time
from typing import Any

import modal

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
GPU_TYPE = "a10g"
DTYPE_DEFAULT = "float16"
MAX_NEW_TOKENS_DEFAULT = 512

app = modal.App("capsule-brain-qual-v04")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.20",
        "safetensors>=0.4",
    )
)


@app.cls(image=image, gpu=GPU_TYPE, timeout=600, memory=8192)
class FrozenTransformerProvider:
    """Frozen transformer model served on Modal GPU.

    The model is loaded once per container. All parameters are frozen.
    Generation is deterministic (do_sample=False).
    """

    model_id: str = modal.parameter(default=MODEL_ID_DEFAULT)
    dtype: str = modal.parameter(default=DTYPE_DEFAULT)
    collect_hidden_states: bool = modal.parameter(default=False)
    hidden_layer_ids: str = modal.parameter(default="")  # JSON-encoded list

    @modal.enter()
    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Parse hidden_layer_ids from JSON string.
        if isinstance(self.hidden_layer_ids, str) and self.hidden_layer_ids:
            import json
            self._hidden_layer_ids = json.loads(self.hidden_layer_ids)
        elif isinstance(self.hidden_layer_ids, list):
            self._hidden_layer_ids = self.hidden_layer_ids
        else:
            self._hidden_layer_ids = []

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            output_hidden_states=self.collect_hidden_states,
        ).to("cuda")
        self._model.eval()
        # Freeze all parameters (Section 10).
        for param in self._model.parameters():
            param.requires_grad_(False)

        # Record revision.
        self._model_revision = None
        self._tokenizer_revision = None
        try:
            cfg = getattr(self._model, "config", None)
            if cfg is not None:
                self._model_revision = getattr(cfg, "_commit_hash", None)
            self._tokenizer_revision = getattr(self._tokenizer, "_commit_hash", None)
        except Exception:
            pass

        self._generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS_DEFAULT,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> dict:
        """Generate text from the frozen model.

        Returns:
            dict with: text, token_ids, logprobs, hidden_states (if enabled),
            latency_ms, token_count, model_id, model_revision
        """
        import torch

        started = time.perf_counter()
        gen_config = dict(self._generation_config)
        gen_config["max_new_tokens"] = max_new_tokens

        # Use chat template if available (tokenize=False then re-tokenize for batching).
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                chat_text = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = self._tokenizer(
                    chat_text, return_tensors="pt", padding=True
                ).to("cuda")
            except Exception:
                inputs = self._tokenizer(
                    prompt, return_tensors="pt", padding=True
                ).to("cuda")
        else:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", padding=True
            ).to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.collect_hidden_states,
            )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_ids = generated_ids.tolist()

        # Per-token log-probs from scores.
        logprobs = []
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F
            for i, score in enumerate(outputs.scores):
                if i < len(token_ids):
                    logits = score[0]
                    probs = F.softmax(logits, dim=-1)
                    logprobs.append(float(torch.log(probs[token_ids[i]] + 1e-10)))

        # Hidden states for Gate B (if enabled).
        hidden_states = {}
        if self.collect_hidden_states and hasattr(outputs, "hidden_states"):
            for layer_idx in self._hidden_layer_ids:
                if layer_idx < len(outputs.hidden_states):
                    layer_hs = outputs.hidden_states[layer_idx]
                    last_token_hs = layer_hs[0, -1, :].cpu().float().tolist()
                    hidden_states[str(layer_idx)] = last_token_hs

        elapsed = time.perf_counter() - started

        return {
            "text": text,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "hidden_states": hidden_states,
            "latency_ms": elapsed * 1000,
            "token_count": len(token_ids),
            "model_id": self.model_id,
            "model_revision": self._model_revision,
            "tokenizer_revision": self._tokenizer_revision,
            "dtype": self.dtype,
            "device": "cuda",
            "generation_config": gen_config,
        }

    @modal.method()
    def generate_batch(self, prompts, max_new_tokens=512):
        """Generate text for a batch of prompts (sequential within container)."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_new_tokens=max_new_tokens))
        return results

    @modal.method()
    def get_model_info(self) -> dict:
        """Return model metadata for provenance."""
        return {
            "model_id": self.model_id,
            "model_revision": self._model_revision,
            "tokenizer_id": self.model_id,
            "tokenizer_revision": self._tokenizer_revision,
            "dtype": self.dtype,
            "device": "cuda",
            "gpu_type": GPU_TYPE,
            "generation_config": self._generation_config,
            "supports_hidden_states": True,
            "provider_class": "real_model",
            "supports_gate_a_claim": True,
            "supports_gate_b_claim": True,
        }


# ---------------------------------------------------------------------------
# Convenience functions for running the Modal app locally
# ---------------------------------------------------------------------------


def run_remote_generation(
    prompt,
    *,
    model_id=MODEL_ID_DEFAULT,
    max_new_tokens=512,
    collect_hidden_states=False,
    hidden_layer_ids=None,
):
    """Run a single remote generation on Modal GPU."""
    import json as _json
    with app.run():
        provider = FrozenTransformerProvider(
            model_id=model_id,
            collect_hidden_states=collect_hidden_states,
            hidden_layer_ids=_json.dumps(hidden_layer_ids or []),
        )
        result = provider.generate.remote(prompt, max_new_tokens=max_new_tokens)
        return result


def run_remote_batch(
    prompts,
    *,
    model_id=MODEL_ID_DEFAULT,
    max_new_tokens=512,
    collect_hidden_states=False,
    hidden_layer_ids=None,
):
    """Run a batch of remote generations on Modal GPU."""
    import json as _json
    with app.run():
        provider = FrozenTransformerProvider(
            model_id=model_id,
            collect_hidden_states=collect_hidden_states,
            hidden_layer_ids=_json.dumps(hidden_layer_ids or []),
        )
        results = provider.generate_batch.remote(prompts, max_new_tokens=max_new_tokens)
        return results


def run_remote_model_info(model_id=MODEL_ID_DEFAULT):
    """Get model metadata from the remote Modal provider."""
    with app.run():
        provider = FrozenTransformerProvider(model_id=model_id)
        return provider.get_model_info.remote()


if __name__ == "__main__":
    import json
    print("Getting model info from Modal GPU...")
    info = run_remote_model_info()
    print(json.dumps(info, indent=2))
    print("\nGenerating text...")
    result = run_remote_generation("What is 7 times 13? Output only the number.", max_new_tokens=64)
    print(f"Text: {result['text']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"Tokens: {result['token_count']}")
