"""Phase 4: KV cache footprint across attention variants (MHA vs GQA).

Compares gpt2 (pure multi-head attention) and Qwen2.5-0.5B (grouped-query
attention). For each model we read the head/layer dimensions, compute the KV
cache bytes per token, and report the GQA reduction factor — the per-token
cache shrinkage versus a same-dimension MHA model.

Output: results/data/attention_variants.csv  +  a printed comparison table.
Run: ``python experiments/phase4_attention.py``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.attention_variants import (
    get_model_dims,
    kv_bytes_per_token,
    mha_equivalent_bytes_per_token,
)


def build_table() -> pd.DataFrame:
    rows = []
    for model_name in [config.MHA_MODEL, config.GQA_MODEL]:
        dims = get_model_dims(model_name)
        actual = kv_bytes_per_token(dims)
        mha_equiv = mha_equivalent_bytes_per_token(dims)
        rows.append({
            "model": model_name,
            "attention_type": dims.attention_type,
            "num_hidden_layers": dims.num_hidden_layers,
            "num_attention_heads": dims.num_attention_heads,
            "num_key_value_heads": dims.num_key_value_heads,
            "head_dim": dims.head_dim,
            "kv_bytes_per_token": actual,
            "mha_equiv_bytes_per_token": mha_equiv,
            "gqa_reduction_factor": round(dims.gqa_reduction_factor, 3),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print("\n--- KV cache per-token footprint by attention variant ---")
    show = df.copy()
    show["kv_KiB_per_token"] = (show["kv_bytes_per_token"] / 1024).round(2)
    cols = ["model", "attention_type", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "kv_bytes_per_token",
            "kv_KiB_per_token", "gqa_reduction_factor"]
    print(show[cols].to_string(index=False))

    print("\n--- Acceptance: GQA reduction vs same-dimension MHA ---")
    for _, r in df.iterrows():
        factor = r["mha_equiv_bytes_per_token"] / r["kv_bytes_per_token"]
        print(f"  {r['model']:30s} ({r['attention_type']}): "
              f"{r['kv_bytes_per_token']:6d} B/tok  vs  MHA-equivalent "
              f"{r['mha_equiv_bytes_per_token']:6d} B/tok  -> {factor:.2f}x reduction "
              f"(head ratio {r['num_attention_heads']}/{r['num_key_value_heads']})")


def main() -> None:
    df = build_table()
    df.to_csv(config.ATTENTION_CSV, index=False)
    report(df)
    print(f"\nWrote {config.ATTENTION_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
