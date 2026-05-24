"""Phase 2: latency benchmark across the sequence-length grid.

For each starting context length L in the grid and each cache mode, run a
manual decode loop of DECODE_TOKENS steps, with one discarded warmup iteration
and NUM_REPEATS measured iterations aggregated by median per decode step.
Separately measure prefill time as a function of prompt length.

Outputs:
  results/data/latency.csv  -> mode, sequence_length, step_index, step_latency_ms
  results/data/prefill.csv  -> prompt_length, prefill_ms

Run: ``python experiments/phase2_latency.py``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import benchmark, config
from src.model_loader import load_model

MODES = {"cache_on": True, "cache_off": False}


def run_latency(model, vocab_size) -> pd.DataFrame:
    rows = []
    for mode_name, use_cache in MODES.items():
        for L in config.SEQUENCE_LENGTHS:
            # Discarded warmup.
            for _ in range(config.WARMUP_ITERS):
                ids = benchmark.make_input_ids(L, vocab_size)
                benchmark.decode_loop_latencies(
                    model, ids, config.DECODE_TOKENS, use_cache)

            # Measured repeats: shape (repeats, num_steps).
            reps = []
            for _ in range(config.NUM_REPEATS):
                ids = benchmark.make_input_ids(L, vocab_size)
                reps.append(benchmark.decode_loop_latencies(
                    model, ids, config.DECODE_TOKENS, use_cache))
            median_per_step = np.median(np.array(reps), axis=0)

            for step_index, lat in enumerate(median_per_step):
                rows.append({
                    "mode": mode_name,
                    "sequence_length": L,
                    "step_index": step_index,
                    "step_latency_ms": float(lat),
                })
            print(f"  {mode_name:9s} L={L:5d}  "
                  f"median step latency = {np.median(median_per_step):7.3f} ms")
    return pd.DataFrame(rows)


def run_prefill(model, vocab_size) -> pd.DataFrame:
    rows = []
    for L in config.SEQUENCE_LENGTHS:
        for _ in range(config.WARMUP_ITERS):
            benchmark.time_prefill(model, benchmark.make_input_ids(L, vocab_size))
        reps = [benchmark.time_prefill(model, benchmark.make_input_ids(L, vocab_size))
                for _ in range(config.NUM_REPEATS)]
        ms = float(np.median(reps))
        rows.append({"prompt_length": L, "prefill_ms": ms})
        print(f"  prefill L={L:5d}  {ms:7.3f} ms")
    return pd.DataFrame(rows)


def report_acceptance(latency_df: pd.DataFrame) -> None:
    """Print the smallest- vs largest-L per-step latency for each mode."""
    print("\n--- Acceptance summary (median per-step latency by start length) ---")
    Lmin, Lmax = min(config.SEQUENCE_LENGTHS), max(config.SEQUENCE_LENGTHS)
    for mode_name in MODES:
        sub = latency_df[latency_df["mode"] == mode_name]
        lo = sub[sub["sequence_length"] == Lmin]["step_latency_ms"].median()
        hi = sub[sub["sequence_length"] == Lmax]["step_latency_ms"].median()
        print(f"  {mode_name:9s}  L={Lmin}: {lo:7.3f} ms   "
              f"L={Lmax}: {hi:7.3f} ms   ratio={hi / lo:5.2f}x")


def main() -> None:
    print(f"Loading {config.DEFAULT_MODEL} ...")
    tokenizer, model = load_model(config.DEFAULT_MODEL)
    vocab_size = model.config.vocab_size

    print("\nDecode-loop latency:")
    latency_df = run_latency(model, vocab_size)
    latency_df.to_csv(config.LATENCY_CSV, index=False)
    print(f"Wrote {config.LATENCY_CSV} ({len(latency_df)} rows)")

    print("\nPrefill latency:")
    prefill_df = run_prefill(model, vocab_size)
    prefill_df.to_csv(config.PREFILL_CSV, index=False)
    print(f"Wrote {config.PREFILL_CSV} ({len(prefill_df)} rows)")

    report_acceptance(latency_df)


if __name__ == "__main__":
    main()
