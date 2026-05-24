# KV Cache Mechanics — Benchmark Project

## Goal
Empirically measure the effect of KV caching on transformer inference. Reproduce two known results on a local model: (1) decode latency scales quadratically with sequence length when caching is disabled and linearly when enabled, (2) KV cache memory grows linearly with sequence length. Extend to a multi-head vs grouped-query attention cache-size comparison.

## Hardware and environment constraints — READ FIRST
- Target machine: remote Linux pod on RunPod with an NVIDIA RTX 4000 Ada GPU. Device backend is CUDA. Use `device = "cuda"`.
- Verify the GPU before any work: run `nvidia-smi` and confirm a GPU is listed.
- Do NOT create a virtual environment and do NOT install or reinstall torch. The RunPod PyTorch template provides torch with CUDA in the system Python. Use the system Python directly. Required extra packages are already installed: transformers, accelerate, numpy, pandas, matplotlib, tqdm, pytest.
- Memory measurement uses CUDA APIs. Before a measured interval call `torch.cuda.reset_peak_memory_stats()`. After it read `torch.cuda.max_memory_allocated()` for the exact peak and `torch.cuda.memory_allocated()` for the current value. No manual peak sampler is needed.
- Before any timing measurement call `torch.cuda.synchronize()`. CUDA dispatch is asynchronous; without synchronization timings are meaningless.
- Attention implementation: load models with `attn_implementation="eager"` for all phases so decode-step behavior stays explicit and comparable. Do not switch to sdpa or flash-attention; comparability across phases matters more than raw speed here.
- Models are small and ungated. Default models:
  - Latency/memory phases: `Qwen/Qwen2.5-0.5B-Instruct` (grouped-query attention).
  - Attention-variant phase: `gpt2` (pure multi-head attention) vs `Qwen/Qwen2.5-0.5B-Instruct` (GQA).
- In `src/config.py` set `HF_HOME` to `./hf_cache` so model downloads stay inside the repo and are gitignored.

## Repository layout
src/
config.py             # device, model names, sequence-length grid, output paths, HF_HOME
model_loader.py       # load model + tokenizer, CUDA placement, eager attention
memory.py             # CUDA memory measurement helpers
benchmark.py          # core: manual decode loop, cache on/off, per-step timing
attention_variants.py # KV cache tensor-shape and byte-size computation per model
plotting.py           # matplotlib figures from results CSVs
experiments/
phase1_smoke.py
phase2_latency.py
phase3_memory.py
phase4_attention.py
phase5_analysis.py
results/
data/                 # CSV outputs
plots/                # PNG outputs
tests/
test_smoke.py

## Core experimental design
The latency and memory results require a manual decode loop, not `model.generate()`. Two modes per run:
- cache=True: retain `past_key_values` across steps, feed only the single new token each step.
- cache=False: feed the entire growing sequence each step, discard `past_key_values`.
Time each individual decode step and record `(step_index, sequence_length, step_latency_ms)`. Cache=False step latency rises linearly with position, making cumulative cost quadratic. Cache=True step latency stays roughly flat.
Separately measure prefill time by varying initial prompt length and timing the first forward pass.

## Phase tracker
Update this block at the end of every phase. Mark the completed phase DONE, write a one-line result summary, set the next phase to IN PROGRESS.

- [ ] Phase 1 — Scaffolding and smoke test — STATUS: NOT STARTED
- [ ] Phase 2 — Latency benchmark — STATUS: NOT STARTED
- [ ] Phase 3 — Memory benchmark — STATUS: NOT STARTED
- [ ] Phase 4 — Attention variant comparison — STATUS: NOT STARTED
- [ ] Phase 5 — Analysis and writeup — STATUS: NOT STARTED

## Phase definitions

### Phase 1 — Scaffolding and smoke test
Build: `src/config.py`, `src/model_loader.py`, `src/memory.py`, `experiments/phase1_smoke.py`, `tests/test_smoke.py`.
- `config.py` holds the device string, model identifiers, the sequence-length grid (e.g. [32, 64, 128, 256, 512, 1024]), decode-token count, output directory paths, and sets HF_HOME.
- `model_loader.py` loads tokenizer and model in fp16 onto CUDA with `attn_implementation="eager"`, returns both.
- `memory.py` provides helpers wrapping `reset_peak_memory_stats`, `max_memory_allocated`, `memory_allocated`, `empty_cache`.
- `phase1_smoke.py` loads the default model, runs a short `model.generate()`, prints output text, prints whether `past_key_values` is returned, prints the device of model parameters.
- `test_smoke.py` asserts CUDA is available, asserts the model loads, asserts a forward pass returns logits of expected shape.
Acceptance: `python experiments/phase1_smoke.py` prints coherent generated text. `pytest` passes.

### Phase 2 — Latency benchmark
Build: `src/benchmark.py`, `experiments/phase2_latency.py`.
- `benchmark.py` implements the manual decode loop described in Core experimental design, parameterized by cache on/off, plus a separate prefill timing function. All timing wrapped with `torch.cuda.synchronize()`.
- `phase2_latency.py` runs the decode loop for both modes across the sequence-length grid, at least 3 repeats per point with median aggregation, plus one discarded warmup iteration.
- Output: `results/data/latency.csv` with columns `mode, sequence_length, step_index, step_latency_ms` and `results/data/prefill.csv` with columns `prompt_length, prefill_ms`.
Acceptance: CSVs populated. Cache=False per-step latency visibly increases with sequence length; cache=True does not.

### Phase 3 — Memory benchmark
Build: `experiments/phase3_memory.py`.
- For cache=True, wrap each decode interval with `reset_peak_memory_stats()` then read `max_memory_allocated()`. Record the peak per sequence length.
- Compute the theoretical KV cache size: `2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes`. Compare measured vs theoretical.
- Output: `results/data/memory.csv` with columns `sequence_length, measured_mb, theoretical_mb`.
Acceptance: measured memory grows approximately linearly with sequence length; measured and theoretical curves are the same order of magnitude.

### Phase 4 — Attention variant comparison
Build: `src/attention_variants.py`, `experiments/phase4_attention.py`.
- For `gpt2` (MHA) and `Qwen2.5-0.5B` (GQA), extract `num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, `head_dim` from each config.
- Compute KV cache bytes per token for each model. Show the GQA reduction factor `num_attention_heads / num_key_value_heads`.
- Output: `results/data/attention_variants.csv` and a printed comparison table.
Acceptance: GQA model shows a KV cache per-token footprint reduced by the head-ratio factor relative to a same-dimension MHA model.

### Phase 5 — Analysis and writeup
Build: `src/plotting.py`, `experiments/phase5_analysis.py`, populate `README.md`.
- Plots: (1) per-step latency vs sequence length, both modes; (2) cumulative decode time vs tokens generated, both modes; (3) measured vs theoretical KV cache memory; (4) bar chart of KV cache bytes per token across attention variants.
- Save PNGs to `results/plots/`.
- `README.md`: project description, the compute-vs-memory tradeoff conclusion, the four figures embedded, reproduction instructions, and a short paragraph connecting the results to agent-system design (stable prefix, append-only context, prefix caching economics).
Acceptance: all four PNGs generated, README complete and renders correctly.

## Working protocol for Claude Code
- Work one phase at a time. Do not start a phase before the previous one is marked DONE in the phase tracker.
- At the start of a phase, read the phase tracker and the relevant phase definition.
- At the end of a phase: run the acceptance check, then edit the phase tracker block in this file — mark the phase DONE with a one-line result, set the next phase to IN PROGRESS. Commit with message `phaseN: <summary>` and push.
- Keep all measurement code synchronization-correct on CUDA. Any timing without `torch.cuda.synchronize()` is a bug.
- Do not create a virtual environment. Do not install or reinstall torch. Do not add gated models.
