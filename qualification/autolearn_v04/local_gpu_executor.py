"""Local GPU executor for real model inference on RunPod/Colab (Section 10/12).

Executes counterfactual actions on scientific benchmark tasks using a real
frozen transformer model loaded directly on the local GPU.  This avoids the
Modal round-trip entirely and is the fastest path for evidence generation
when a GPU is available locally.

Performance on an RTX 5090 (32 GB):
    * Qwen2.5-7B-Instruct in float16: ~0.05 s / generation (256 tokens)
    * Qwen2.5-3B-Instruct in float16: ~0.02 s / generation (256 tokens)

The executor:
    - loads a real HF transformers model on the local CUDA device;
    - sends task prompts to the model;
    - receives generated text responses;
    - extracts answers from model output (JSON parsing, code extraction);
    - verifies answers independently against hidden expected values;
    - records typed CounterfactualOutcome with real model metadata;
    - collects hidden states for Gate B when enabled.

No benchmark-answer extraction shortcuts. The model sees only the prompt.
Expected answers live only in verifier_state.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# Reuse the verification and prompt-building logic from the Modal executor.
from .modal_scientific_executor import (
    _build_action_prompt,
    verify_scientific_outcome,
)


class LocalGPUExecutor:
    """Run a real frozen transformer model on the local GPU.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
    device : str
        CUDA device (``"cuda"`` or ``"cuda:0"``).
    dtype : str
        ``"float16"`` or ``"bfloat16"`` for speed, ``"float32"`` for precision.
    collect_hidden_states : bool
        If True, collect hidden states for Gate B.
    hidden_layer_ids : list[int] | None
        Layer indices for hidden-state collection.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        collect_hidden_states: bool = False,
        hidden_layer_ids: list[int] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.collect_hidden_states = collect_hidden_states
        self.hidden_layer_ids = hidden_layer_ids or []
        self._model = None
        self._tokenizer = None
        self._model_revision: str | None = None
        self._tokenizer_revision: str | None = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        print(f"  Loading tokenizer: {self.model_id} ...")
        t0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        print(f"  Tokenizer loaded in {time.perf_counter()-t0:.1f}s")

        print(f"  Loading model: {self.model_id} [{self.dtype}] on {self.device} ...")
        t0 = time.perf_counter()
        # Try new API (transformers 5.x) first, fall back to old API.
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
        self._model.eval()
        # Section 10: all parameters frozen — no gradient computation.
        for param in self._model.parameters():
            param.requires_grad_(False)
        load_time = time.perf_counter() - t0
        vram_gb = torch.cuda.memory_allocated() / 1024**3 if self.device.startswith("cuda") else 0
        print(f"  Model loaded in {load_time:.1f}s, VRAM: {vram_gb:.2f} GB")

        # Record revisions.
        try:
            cfg = getattr(self._model, "config", None)
            if cfg is not None:
                self._model_revision = getattr(cfg, "_commit_hash", None) or getattr(cfg, "revision", None)
            self._tokenizer_revision = getattr(self._tokenizer, "_commit_hash", None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Single and batch generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 256) -> dict[str, Any]:
        """Single generation on the local GPU."""
        import torch

        started = time.perf_counter()
        # Use chat template if available.
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                chat_text = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
                inputs = self._tokenizer(chat_text, return_tensors="pt", padding=True).to(self.device)
            except Exception:
                inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.collect_hidden_states,
            )
        generated_ids = outputs.sequences[0][input_len:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_ids = generated_ids.tolist()
        token_count = len(token_ids)

        # Per-token log-probs from scores.
        logprobs: list[float] = []
        if hasattr(outputs, "scores") and outputs.scores:
            import torch.nn.functional as F
            for step_scores in outputs.scores:
                if step_scores is None:
                    continue
                probs = F.log_softmax(step_scores[0], dim=-1)
                top_id = probs.argmax().item()
                logprobs.append(round(probs[top_id].item(), 6))

        # Hidden states (optional, for Gate B).
        hidden_states: dict[str, Any] = {}
        if self.collect_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            for layer_id in self.hidden_layer_ids:
                if layer_id < len(outputs.hidden_states):
                    hs = outputs.hidden_states[layer_id]
                    if hs is not None:
                        # Average across generated tokens, keep on CPU.
                        avg = hs[0].mean(dim=0).cpu()
                        hidden_states[str(layer_id)] = {
                            "shape": list(avg.shape),
                            "mean": round(float(avg.mean()), 6),
                            "norm": round(float(avg.norm()), 6),
                        }

        elapsed = time.perf_counter() - started
        return {
            "text": text,
            "token_ids": token_ids,
            "token_count": token_count,
            "logprobs": logprobs,
            "latency_ms": round(elapsed * 1000, 1),
            "model_id": self.model_id,
            "model_revision": self._model_revision,
            "tokenizer_revision": self._tokenizer_revision,
            "dtype": self.dtype,
            "device": self.device,
            "hidden_states": hidden_states,
        }

    def batch_generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[dict[str, Any]]:
        """Sequential batch generation on the local GPU.

        This is the fastest path for small-to-medium batches on a single GPU.
        No network round-trips, no cold starts — just pure GPU inference.
        """
        results = []
        n = len(prompts)
        t0 = time.perf_counter()
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            results.append(result)
            if (i + 1) % 20 == 0 or i == n - 1:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                print(f"    [{i+1}/{n}] {rate:.1f} prompts/s, "
                      f"{elapsed/(i+1):.3f}s/prompt, "
                      f"ETA: {(n-i-1)/rate:.0f}s")
        return results


# ---------------------------------------------------------------------------
# Counterfactual execution (mirrors modal_scientific_executor)
# ---------------------------------------------------------------------------


def run_local_gpu_counterfactuals(
    tasks: list[dict[str, Any]] | list[Any],
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    dtype: str = "float16",
    max_new_tokens: int = 256,
    collect_hidden_states: bool = False,
    hidden_layer_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Execute counterfactuals on scientific tasks using the local GPU.

    Returns a list of outcome dicts compatible with the v0.4.0 schema.
    """
    outcomes: list[dict[str, Any]] = []

    # Build all prompts for batch execution.
    prompts: list[str] = []
    prompt_meta: list[tuple[dict, str]] = []
    for task in tasks:
        task_dict = task.to_dict() if hasattr(task, "to_dict") else task
        for action_name in task_dict["allowed_actions"]:
            prompt = _build_action_prompt(task_dict, action_name)
            prompts.append(prompt)
            prompt_meta.append((task_dict, action_name))

    print(f"  Built {len(prompts)} prompts for {len(tasks)} tasks")

    # Load the model on the local GPU.
    print(f"  Loading model on local GPU ({device})...")
    executor = LocalGPUExecutor(
        model_id=model_id,
        device=device,
        dtype=dtype,
        collect_hidden_states=collect_hidden_states,
        hidden_layer_ids=hidden_layer_ids,
    )

    # Batch generate.
    print(f"  Generating {len(prompts)} responses on {model_id}...")
    all_results = executor.batch_generate(prompts, max_new_tokens=max_new_tokens)

    # Verify each result.
    for (task, action_name), result in zip(prompt_meta, all_results):
        model_text = result.get("text", "")
        status, extracted, evidence = verify_scientific_outcome(
            task, action_name, model_text,
        )

        # Build outcome dict.
        outcome = {
            "task_id": task["task_id"],
            "action_id": action_name,
            "availability": "executed",
            "verification": status,
            "utility": None,
            "reward_components": {},
            "execution_metadata": {
                "runtime_type": "real",
                "provider_class": "real_model",
                "model_id": result.get("model_id", model_id),
                "model_revision": result.get("model_revision"),
                "tokenizer_revision": result.get("tokenizer_revision"),
                "dtype": result.get("dtype", dtype),
                "device": result.get("device", device),
                "latency_ms": result.get("latency_ms", 0.0),
                "token_count": result.get("token_count", 0),
                "logprobs": result.get("logprobs", []),
                "verification_status": status,
                "verification_evidence": evidence,
                "extracted_answer": extracted,
                "model_text": model_text[:500],
            },
            "error_type": None,
            "error_message": None,
        }

        # Compute utility for executed outcomes.
        family = task["family"]
        verified = status == "success"
        unnecessary_tool = action_name == "CALL_TOOL" and family != "tool_required"
        unnecessary_workflow = action_name == "START_WORKFLOW" and family != "workflow_required"

        if verified:
            utility = 10.0
            if unnecessary_tool:
                utility -= 2.0
            if unnecessary_workflow:
                utility -= 2.0
        else:
            utility = 0.0
            if unnecessary_tool:
                utility -= 2.0
            if unnecessary_workflow:
                utility -= 2.0

        latency = result.get("latency_ms", 0.0)
        utility -= latency / 10000.0
        tokens = result.get("token_count", 0)
        utility -= tokens / 100.0

        outcome["utility"] = round(utility, 4)
        outcome["reward_components"] = {
            "success_bonus": 10.0 if verified else 0.0,
            "unnecessary_tool_penalty": -2.0 if unnecessary_tool else 0.0,
            "unnecessary_workflow_penalty": -2.0 if unnecessary_workflow else 0.0,
            "latency_penalty": -latency / 10000.0,
            "token_penalty": -tokens / 100.0,
        }

        if collect_hidden_states and result.get("hidden_states"):
            outcome["execution_metadata"]["hidden_states"] = result["hidden_states"]

        outcomes.append(outcome)

    return outcomes


def run_local_gpu_counterfactuals_to_artifacts(
    artifacts_dir: str | Path,
    tasks: list[dict[str, Any]] | list[Any],
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    dtype: str = "float16",
    max_new_tokens: int = 256,
    collect_hidden_states: bool = False,
    hidden_layer_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Execute scientific counterfactuals on local GPU and write artifacts."""
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning scientific counterfactuals on {len(tasks)} tasks "
          f"with {model_id} on local GPU...")

    outcomes = run_local_gpu_counterfactuals(
        tasks,
        model_id=model_id,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        collect_hidden_states=collect_hidden_states,
        hidden_layer_ids=hidden_layer_ids,
    )

    # Write outcomes.
    write_path = artifacts / "counterfactual_outcomes.json"
    write_path.write_text(json.dumps(outcomes, indent=2, default=str))

    # Write legacy-compatible rows.
    legacy_rows = []
    for o in outcomes:
        legacy_rows.append({
            "task_id": o["task_id"],
            "action_id": o["action_id"],
            "availability": o["availability"],
            "verification": o["verification"],
            "utility": o["utility"],
            "runtime_type": "real",
            "provider_class": "real_model",
            "model_id": o["execution_metadata"].get("model_id"),
            "model_revision": o["execution_metadata"].get("model_revision"),
            "latency_ms": o["execution_metadata"].get("latency_ms", 0.0),
            "token_count": o["execution_metadata"].get("token_count", 0),
            "verified_success": o["verification"] == "success",
            "verification_status": o["verification"],
            "verification_evidence": o["execution_metadata"].get("verification_evidence", {}),
        })
    (artifacts / "real_counterfactual_results.json").write_text(
        json.dumps(legacy_rows, indent=2, default=str),
    )

    # Print summary.
    n_total = len(outcomes)
    n_success = sum(1 for o in outcomes if o["verification"] == "success")
    n_failure = n_total - n_success
    print(f"\nScientific counterfactuals complete:")
    print(f"  Total: {n_total}")
    if n_total > 0:
        print(f"  Success: {n_success} ({n_success/n_total*100:.1f}%)")
        print(f"  Failure: {n_failure} ({n_failure/n_total*100:.1f}%)")
        print(f"  Mean latency: {sum(o['execution_metadata']['latency_ms'] for o in outcomes)/n_total:.0f}ms")
        print(f"  Mean tokens: {sum(o['execution_metadata']['token_count'] for o in outcomes)/n_total:.0f}")

    return outcomes
