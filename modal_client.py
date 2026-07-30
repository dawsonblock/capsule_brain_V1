"""Client for the deployed Modal GPU provider.

Talks to the deployed `capsule-brain-qual-v04` app on Modal.
Containers stay warm between calls, so subsequent calls
have no cold start (~0.1s per generation after warmup).

Usage:
    from modal_client import ModalGPIClient
    client = ModalGPIClient()
    result = client.generate("What is 7*13?")
    results = client.batch_generate(["prompt1", "prompt2", ...])
"""
import modal

# Look up the deployed provider class directly (no app.lookup needed).
_FrozenTransformerProvider = modal.Cls.from_name(
    "capsule-brain-qual-v04", "FrozenTransformerProvider",
)
_generate_single_fn = modal.Function.from_name(
    "capsule-brain-qual-v04", "generate_single",
)


class ModalGPIClient:
    """Fast client for the deployed Modal GPU provider.

    The deployed app keeps containers warm, so subsequent calls
    have no cold start (~0.1s per generation instead of ~50s).
    """

    def __init__(
        self,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        collect_hidden_states=False,
        hidden_layer_ids=None,
    ):
        import json
        self.model_id = model_id
        self.collect_hidden_states = collect_hidden_states
        self.hidden_layer_ids_json = json.dumps(hidden_layer_ids or [])
        self._provider = _FrozenTransformerProvider(
            model_id=model_id,
            collect_hidden_states=collect_hidden_states,
            hidden_layer_ids=self.hidden_layer_ids_json,
        )

    def generate(self, prompt, max_new_tokens=256):
        """Single generation on warm container."""
        return self._provider.generate.remote(prompt, max_new_tokens=max_new_tokens)

    def batch_generate(self, prompts, max_new_tokens=256):
        """Batch generation — sequential on one warm container.

        Fastest for small-medium batches. After warmup, ~0.1s per prompt.
        """
        return self._provider.generate_batch.remote(prompts, max_new_tokens=max_new_tokens)

    def parallel_generate(self, prompts, max_new_tokens=256):
        """Parallel generation via .map() across multiple containers.

        Best for large batches (>20 prompts). With warm containers,
        this is both parallel AND cold-start-free.
        """
        kwargs = {
            "model_id": self.model_id,
            "max_new_tokens": max_new_tokens,
            "collect_hidden_states": self.collect_hidden_states,
            "hidden_layer_ids_json": self.hidden_layer_ids_json,
        }
        return list(_generate_single_fn.map(
            prompts,
            kwargs=kwargs,
            order_outputs=True,
        ))

    def model_info(self):
        """Get model metadata."""
        return self._provider.get_model_info.remote()


if __name__ == "__main__":
    import time
    print("Testing deployed Modal GPU client...")
    client = ModalGPIClient()

    # Warmup call.
    print("Warming up container...")
    start = time.time()
    info = client.model_info()
    print(f"  Model: {info['model_id']} rev: {info['model_revision']}")
    print(f"  Warmup: {time.time()-start:.1f}s")

    # Single generation (warm).
    print("\nSingle generation (warm)...")
    start = time.time()
    result = client.generate("What is 7 times 13? Output only the number.", max_new_tokens=64)
    print(f"  Result: {result['text'].strip()}")
    print(f"  Time: {time.time()-start:.1f}s")

    # Batch of 6 (warm container, sequential).
    prompts = [
        "What is 2+2? Output only the number.",
        "What is 3*5? Output only the number.",
        "What is 10-7? Output only the number.",
        "What is 100/4? Output only the number.",
        "What is 7*13? Output only the number.",
        "What is 99+1? Output only the number.",
    ]

    print(f"\nBatch of {len(prompts)} (warm)...")
    start = time.time()
    results = client.batch_generate(prompts, max_new_tokens=32)
    elapsed = time.time() - start
    print(f"  Total: {elapsed:.1f}s, avg: {elapsed/len(prompts):.1f}s/prompt")
    for p, r in zip(prompts, results):
        print(f"    {p[:35]:35s} -> {r['text'].strip()}")
