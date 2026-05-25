# KV Cache Mechanics

An empirical study of how key–value (KV) caching changes transformer inference.
Using a small local model on a single GPU, it reproduces two textbook results
and adds an attention-variant comparison:

1. **Decode latency** scales roughly **linearly** with sequence length when the
   KV cache is enabled, and the *cumulative* cost of decoding grows
   **quadratically** when it is disabled.
2. **KV cache memory** grows **linearly** with sequence length, matching the
   theoretical size to the byte.
3. **Grouped-query attention (GQA)** shrinks the KV cache per token by the
   query-to-KV head ratio versus multi-head attention (MHA).

All measurements use a manual decode loop (not `model.generate()`), eager
attention for explicit and comparable per-step behaviour, and CUDA-correct
timing/memory APIs.

## Setup

- **GPU:** NVIDIA RTX 4000 Ada (20 GB), CUDA backend, fp16 weights.
- **Models:** `Qwen/Qwen2.5-0.5B-Instruct` (GQA) for latency/memory; `gpt2`
  (MHA) vs Qwen for the attention-variant comparison.
- **Attention implementation:** `eager` everywhere — chosen for comparability
  across phases, not raw speed.

## Results

### 1. Decode latency vs sequence length

With the cache **on**, each decode step feeds a single new token and reuses the
stored keys/values, so per-step latency is flat (~40 ms) regardless of context
length. With the cache **off**, every step recomputes attention over the entire
growing sequence, so per-step latency rises with length.

![Per-step decode latency](results/plots/fig1_per_step_latency.png)

| mode | per-step @ L=32 | per-step @ L=1024 | ratio |
|------|-----------------|-------------------|-------|
| cache on  | 40.8 ms | 39.7 ms | **0.97× (flat)** |
| cache off | 41.2 ms | 67.3 ms | **1.63× (rises)** |

The same rise is visible *within a single decode run*. Holding the start fixed
at 1024 tokens and generating 128 more, the cache-off per-step latency climbs
monotonically from ~47 ms to ~72 ms as the re-fed sequence grows, while cache-on
stays flat at ~40 ms. This is the direct measured proof, step by step:

![Per-step latency by decode step](results/plots/fig2_per_step_by_position.png)

Because the cache-off per-step cost grows with position, the **cumulative** cost
of generating *N* tokens grows super-linearly (quadratically), while cache-on
cumulative cost is linear:

![Cumulative decode cost](results/plots/fig3_cumulative_decode.png)

> **Note — Figure 3 is a modeled curve.** It is not a single recorded run: it
> accumulates the measured per-step latency-vs-length grid (Figure 1) over decode
> positions 1…1024, i.e. `cumulative(N) = Σ_{n≤N} per_step(n)`. The per-step
> values are measured; the accumulation is the model. It is shown this way
> because the quadratic only emerges over a large length range, which a single
> 128-step run cannot span.
>
> **Honest caveat.** On a 0.5B model in eager mode this GPU has a ~40 ms fixed
> per-forward floor, so the length-dependent compute term only dominates at
> larger contexts. The cache-on/off contrast and the direction of scaling are
> unambiguous; the absolute curvature is modest at short lengths.

### 2. KV cache memory vs sequence length

The cache stores, per layer, a key and a value tensor of shape
`(num_kv_heads, seq_len, head_dim)`. Its theoretical size is

```
kv_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
```

For Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64, fp16) that is
**12,288 bytes/token**. Measured allocator usage (cache held, model weights
subtracted) matches the theoretical curve exactly and grows linearly:

![KV cache memory](results/plots/fig4_kv_cache_memory.png)

| sequence length | measured | theoretical |
|---|---|---|
| 32   | 0.375 MB | 0.375 MB |
| 512  | 6.000 MB | 6.000 MB |
| 1024 | 12.000 MB | 12.000 MB |

### 3. Attention variants: MHA vs GQA

MHA caches one K/V per attention head; GQA shares each K/V across a group of
query heads, caching only `num_key_value_heads` of them. The per-token cache
therefore shrinks by `num_attention_heads / num_key_value_heads`.

![KV cache per token: MHA vs GQA](results/plots/fig5_attention_variants.png)

| model | attention | layers | attn heads | KV heads | bytes/token |
|---|---|---|---|---|---|
| gpt2 | MHA | 12 | 12 | 12 | 36,864 |
| Qwen2.5-0.5B | GQA | 24 | 14 | 2 | 12,288 |

Qwen's GQA caches **12,288 B/token** versus **86,016 B/token** for an
otherwise-identical MHA model — a **7.00× reduction**, exactly the 14/2 head
ratio.

## The compute–memory tradeoff

The KV cache is a classic space-for-time trade. Without it, generation is
**compute-bound and quadratic**: producing each new token recomputes attention
over all previous tokens, so total work scales with the square of the sequence
length. With it, the cache **trades that quadratic compute for linear compute,
at the cost of a new linear memory term**: you store the keys and values of
every past token (12 KB/token here) so each step attends to them instead of
recomputing them. Quadratic compute is gone; a memory cost that grows linearly
with sequence length appears in its place. GQA then attacks that new memory
term, cutting the per-token footprint by the head ratio with negligible quality
loss — which is why nearly every modern open-weight model ships with it.

## Relevance to agent systems

This is the mechanical basis for several agent-design rules of thumb. Because
the cache is valid only for a **stable prefix**, the cheapest possible context
is one that is **append-only**: keep the system prompt and tool definitions
fixed, add new turns at the end, and never edit earlier content — any change
invalidates the cache from that point and forces a full recompute of everything
after it. **Prefix caching economics** follow directly: cached prefix tokens are
billed and computed at a fraction of fresh tokens, so a long, stable shared
prefix across many requests is far cheaper than the same tokens recomputed each
call. Designing agent context for maximal cache reuse — stable prefix first,
volatile content last — is the same linear-vs-quadratic win measured above,
applied at the level of an entire conversation.

## Reproduction

The project targets a RunPod PyTorch template: torch with CUDA is already in the
system Python. **Do not** create a virtualenv or reinstall torch.

```bash
# Installs transformers==4.46.3 and the other extras; does not touch torch.
pip install -r requirements.txt

nvidia-smi                              # confirm a GPU is present
python experiments/phase1_smoke.py      # load model, generate, inspect cache
python experiments/phase2_latency.py     # -> results/data/latency.csv, prefill.csv
python experiments/phase3_memory.py       # -> results/data/memory.csv
python experiments/phase4_attention.py    # -> results/data/attention_variants.csv
python experiments/phase5_analysis.py      # -> results/plots/*.png
pytest                                       # smoke tests
```

> Note: `transformers` is pinned to `4.46.3` because the 5.x series requires a
> newer torch than the pod's 2.4.1.

## Layout

```
src/
  config.py              device, models, sequence-length grid, paths, HF_HOME
  model_loader.py        fp16 + CUDA + eager-attention model/tokenizer loader
  memory.py              CUDA peak/current memory helpers
  benchmark.py           manual decode loop (cache on/off) + prefill timing
  attention_variants.py  per-model KV cache shape/byte computation
  plotting.py            the five figures
experiments/             phase1_smoke … phase5_analysis
results/data/            CSV outputs (gitignored)
results/plots/           PNG figures (committed)
tests/test_smoke.py      CUDA / load / forward-shape assertions
```
