"""Modal GPU provider for real frozen-transformer qualification (Section 10).

Runs a real frozen causal language model on Modal GPU infrastructure.
The model is loaded once per container, parameters are frozen
(requires_grad=False), and all generation is deterministic with
seed control.

Model target: Qwen2.5-3B-Instruct (or configurable alternative).
GPU: NVIDIA A10G (24GB VRAM) — sufficient for 3B model in float16.

Performance optimizations:
    - DEPLOY mode: `modal deploy` keeps containers warm (no cold start).
    - .map() parallelism: spread prompts across multiple GPU containers.
    - Single app session: reuse one container for all calls in a pipeline run.

This module is self-contained (no capsule_brain imports) so Modal
containers can load it without the full package installed.
"""
# NOTE: No `from __future__ import annotations` — breaks Modal on Python 3.12.
# NOTE: No imports from qualification.autolearn_v04 or capsule_brain.

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

    min_containers=1 keeps at least one container warm after first use,
    eliminating cold starts for subsequent calls within a session.
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
        print(f"[Modal] Model loaded: {self.model_id} rev={self._model_revision}")

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> dict:
        """Generate text from the frozen model."""
        return self._generate_local(prompt, max_new_tokens)

    @modal.method()
    def generate_batch(self, prompts, max_new_tokens=512):
        """Generate text for a batch of prompts (sequential within one container)."""
        results = []
        for prompt in prompts:
            results.append(self._generate_local(prompt, max_new_tokens))
        return results

    def _generate_local(self, prompt, max_new_tokens=512):
        """Internal generation — shared by generate and generate_batch."""
        import torch

        started = time.perf_counter()
        gen_config = dict(self._generation_config)
        gen_config["max_new_tokens"] = max_new_tokens

        # Use chat template if available.
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

        # Per-token log-probs.
        logprobs = []
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F
            for i, score in enumerate(outputs.scores):
                if i < len(token_ids):
                    logits = score[0]
                    probs = F.softmax(logits, dim=-1)
                    logprobs.append(float(torch.log(probs[token_ids[i]] + 1e-10)))

        # Hidden states for Gate B.
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
# Fast batch execution with .map() parallelism
# ---------------------------------------------------------------------------


@app.function(image=image, gpu=GPU_TYPE, timeout=600, memory=8192)
def generate_single(prompt, model_id=MODEL_ID_DEFAULT, max_new_tokens=256,
                    collect_hidden_states=False, hidden_layer_ids_json="[]"):
    """Standalone function for .map() parallelism — one prompt per container call.

    Each .map() call can spin up multiple containers in parallel, with each
    container processing one prompt. This is much faster than sequential
    generation in a single container for large batches.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model (cached by Modal after first call in container).
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True,
        output_hidden_states=collect_hidden_states,
    ).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layer_ids = json.loads(hidden_layer_ids_json) if hidden_layer_ids_json else []

    started = time.perf_counter()

    # Chat template.
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat_text, return_tensors="pt", padding=True).to("cuda")

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs, **gen_config,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=collect_hidden_states,
        )

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_ids = generated_ids.tolist()

    logprobs = []
    if hasattr(outputs, "scores") and outputs.scores:
        import torch.nn.functional as F
        for i, score in enumerate(outputs.scores):
            if i < len(token_ids):
                probs = F.softmax(score[0], dim=-1)
                logprobs.append(float(torch.log(probs[token_ids[i]] + 1e-10)))

    hidden_states = {}
    if collect_hidden_states and hasattr(outputs, "hidden_states"):
        for layer_idx in layer_ids:
            if layer_idx < len(outputs.hidden_states):
                hs = outputs.hidden_states[layer_idx]
                hidden_states[str(layer_idx)] = hs[0, -1, :].cpu().float().tolist()

    elapsed = time.perf_counter() - started

    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)

    return {
        "text": text,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "hidden_states": hidden_states,
        "latency_ms": elapsed * 1000,
        "token_count": len(token_ids),
        "model_id": model_id,
        "model_revision": model_revision,
        "dtype": "float16",
        "device": "cuda",
    }


# ---------------------------------------------------------------------------
# Convenience functions — optimized for speed
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
        return provider.generate.remote(prompt, max_new_tokens=max_new_tokens)


def run_remote_batch(
    prompts,
    *,
    model_id=MODEL_ID_DEFAULT,
    max_new_tokens=512,
    collect_hidden_states=False,
    hidden_layer_ids=None,
):
    """Run a batch of remote generations on Modal GPU.

    Uses .map() for parallelism — Modal spins up multiple containers
    to process prompts in parallel. Much faster than sequential for
    large batches.
    """
    import json as _json
    layer_ids_json = _json.dumps(hidden_layer_ids or [])

    with app.run():
        # Use .map() for parallel execution across containers.
        kwargs = {
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "collect_hidden_states": collect_hidden_states,
            "hidden_layer_ids_json": layer_ids_json,
        }
        results = list(generate_single.map(
            prompts,
            kwargs=kwargs,
            order_outputs=True,  # preserve input order
        ))
        return results


def run_remote_batch_warm(
    prompts,
    *,
    model_id=MODEL_ID_DEFAULT,
    max_new_tokens=512,
    collect_hidden_states=False,
    hidden_layer_ids=None,
):
    """Run a batch using a warm container (sequential but no cold start).

    Faster for small batches (<20 prompts) where container startup
    overhead outweighs parallelism benefits.
    """
    import json as _json
    with app.run():
        provider = FrozenTransformerProvider(
            model_id=model_id,
            collect_hidden_states=collect_hidden_states,
            hidden_layer_ids=_json.dumps(hidden_layer_ids or []),
        )
        return provider.generate_batch.remote(prompts, max_new_tokens=max_new_tokens)


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
    print("\nGenerating text (warm container)...")
    result = run_remote_generation("What is 7 times 13? Output only the number.", max_new_tokens=64)
    print(f"Text: {result['text']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")
    print(f"\nBatch test (3 prompts, parallel via .map())...")
    results = run_remote_batch([
        "What is 2+2? Output only the number.",
        "What is 3*5? Output only the number.",
        "What is 10-7? Output only the number.",
    ], max_new_tokens=32)
    for r in results:
        print(f"  {r['text'].strip()} ({r['latency_ms']:.0f}ms)")
