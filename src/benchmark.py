"""Core decode-loop benchmark: per-step latency with KV cache on or off.

Two modes (see CLAUDE.md "Core experimental design"):
  - cache on : prefill once, then feed a single new token per step and retain
               ``past_key_values`` across steps.
  - cache off: feed the entire growing sequence every step and discard the
               cache, so each step recomputes attention over all positions.

Both modes are aligned by decode position: at ``step_index = s`` and starting
context length ``L`` both produce the token at absolute position ``L + s``.

Every timed interval is bracketed by ``torch.cuda.synchronize()`` because CUDA
dispatch is asynchronous — without it the wall-clock timings are meaningless.
"""
import time

import torch

from . import config


def _sync() -> None:
    torch.cuda.synchronize()


def make_input_ids(length: int, vocab_size: int) -> torch.Tensor:
    """A (1, length) tensor of valid token ids. Content is irrelevant to timing."""
    return torch.randint(0, vocab_size, (1, length), device=config.DEVICE)


@torch.no_grad()
def decode_loop_latencies(model, input_ids: torch.Tensor, num_steps: int,
                          use_cache: bool) -> list[float]:
    """Run a manual decode loop, returning per-step latency in ms.

    The returned list has length ``num_steps``; entry ``s`` is the latency of
    producing the token at absolute position ``input_ids.shape[1] + s``.
    """
    latencies: list[float] = []

    if use_cache:
        # Prefill once (not timed here — prefill is measured separately).
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(num_steps):
            _sync()
            t0 = time.perf_counter()
            out = model(input_ids=next_token, past_key_values=past,
                        use_cache=True)
            _sync()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
    else:
        seq = input_ids
        for _ in range(num_steps):
            _sync()
            t0 = time.perf_counter()
            out = model(input_ids=seq, use_cache=False)
            _sync()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            seq = torch.cat([seq, next_token], dim=1)

    return latencies


@torch.no_grad()
def time_prefill(model, input_ids: torch.Tensor) -> float:
    """Time a single prefill forward pass over ``input_ids``, in ms."""
    _sync()
    t0 = time.perf_counter()
    model(input_ids=input_ids, use_cache=True)
    _sync()
    return (time.perf_counter() - t0) * 1000.0
