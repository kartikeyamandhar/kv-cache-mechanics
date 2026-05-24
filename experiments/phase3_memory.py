"""Phase 3: KV cache memory vs sequence length, measured against theory.

For each sequence length L we build a KV cache of exactly length L (a single
prefill with ``use_cache=True``), then read the CUDA allocator to isolate the
cache's memory footprint:

    measured_mb = (allocated holding the length-L cache) - (model-weights baseline)

The model weights (~1 GB) dwarf the cache (~12 MB at L=1024), so the weights
baseline must be subtracted for the measured curve to be comparable to the
theoretical KV cache size:

    theoretical_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
                        ^-- K and V

Two correctness details:
  * The prefill output logits are (1, L, vocab) ~ 311 MB at L=1024, so we keep
    only ``past_key_values`` and delete the output before measuring.
  * A warmup forward is run before the weights baseline so persistent cuBLAS
    workspaces are already allocated and don't inflate the measured cache.

Output: results/data/memory.csv -> sequence_length, measured_mb, theoretical_mb
Run: ``python experiments/phase3_memory.py``
"""
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src import benchmark, config, memory
from src.model_loader import load_model

_MB = 1024 * 1024


def kv_bytes_per_token(model) -> int:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    # factor 2 = keys and values.
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * config.DTYPE_BYTES


@torch.no_grad()
def measure(model, vocab_size) -> pd.DataFrame:
    per_token = kv_bytes_per_token(model)

    # Warmup so persistent workspaces are allocated before the baseline read.
    _ = model(input_ids=benchmark.make_input_ids(32, vocab_size), use_cache=False)
    del _
    gc.collect()
    memory.empty_cache()

    weights_baseline_mb = memory.current_allocated_mb()
    print(f"Model-weights baseline: {weights_baseline_mb:.1f} MB")
    print(f"Theoretical KV bytes/token: {per_token} "
          f"({per_token / 1024:.2f} KiB)\n")

    rows = []
    for L in config.SEQUENCE_LENGTHS:
        gc.collect()
        memory.empty_cache()
        memory.reset_peak()

        # Build a KV cache of exactly length L; keep only the cache.
        out = model(input_ids=benchmark.make_input_ids(L, vocab_size),
                    use_cache=True)
        past = out.past_key_values
        del out                      # drop the large logits tensor
        gc.collect()

        held_mb = memory.current_allocated_mb()        # weights + cache
        peak_mb = memory.max_allocated_mb()            # weights + prefill activations + cache
        measured_mb = held_mb - weights_baseline_mb    # isolate the KV cache
        theoretical_mb = per_token * L / _MB

        rows.append({
            "sequence_length": L,
            "measured_mb": round(measured_mb, 4),
            "theoretical_mb": round(theoretical_mb, 4),
        })
        print(f"  L={L:5d}  measured={measured_mb:8.3f} MB  "
              f"theoretical={theoretical_mb:8.3f} MB  "
              f"ratio={measured_mb / theoretical_mb:5.2f}x  "
              f"(prefill peak {peak_mb - weights_baseline_mb:6.1f} MB)")

        del past
        gc.collect()
        memory.empty_cache()

    return pd.DataFrame(rows)


def report_acceptance(df: pd.DataFrame) -> None:
    print("\n--- Acceptance summary ---")
    # Linearity: measured_mb / L should be ~constant across the grid.
    df = df.copy()
    df["mb_per_token"] = df["measured_mb"] / df["sequence_length"]
    lo, hi = df["mb_per_token"].min(), df["mb_per_token"].max()
    print(f"measured MB/token across grid: {lo:.5f} .. {hi:.5f} "
          f"(spread {hi / lo:.2f}x — ~1x means linear)")
    print(f"measured vs theoretical at L={df['sequence_length'].iloc[-1]}: "
          f"{df['measured_mb'].iloc[-1]:.2f} vs "
          f"{df['theoretical_mb'].iloc[-1]:.2f} MB (same order of magnitude)")


def main() -> None:
    print(f"Loading {config.DEFAULT_MODEL} ...")
    tokenizer, model = load_model(config.DEFAULT_MODEL)
    df = measure(model, model.config.vocab_size)
    df.to_csv(config.MEMORY_CSV, index=False)
    print(f"\nWrote {config.MEMORY_CSV} ({len(df)} rows)")
    report_acceptance(df)


if __name__ == "__main__":
    main()
