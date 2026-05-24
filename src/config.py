"""Central configuration: device, models, sequence-length grid, output paths.

HF_HOME is set here, at import time, so that any module importing config gets
model downloads redirected into the repo-local (gitignored) cache directory.
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HF_HOME = REPO_ROOT / "hf_cache"
DATA_DIR = REPO_ROOT / "results" / "data"
PLOTS_DIR = REPO_ROOT / "results" / "plots"

# Redirect HuggingFace caches into the repo before transformers is imported.
os.environ.setdefault("HF_HOME", str(HF_HOME))

for _d in (HF_HOME, DATA_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Device ----------------------------------------------------------------
DEVICE = "cuda"

# --- Models ----------------------------------------------------------------
# Latency / memory phases use a small grouped-query-attention model.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Attention-variant phase compares pure multi-head attention vs GQA.
MHA_MODEL = "gpt2"
GQA_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# --- Experimental grid -----------------------------------------------------
SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024]
DECODE_TOKENS = 128          # tokens generated per decode-loop run
NUM_REPEATS = 3              # measured repeats per point (median aggregated)
WARMUP_ITERS = 1             # discarded warmup iterations

# fp16 weights/activations throughout.
DTYPE_BYTES = 2

# --- Output files ----------------------------------------------------------
LATENCY_CSV = DATA_DIR / "latency.csv"
PREFILL_CSV = DATA_DIR / "prefill.csv"
MEMORY_CSV = DATA_DIR / "memory.csv"
ATTENTION_CSV = DATA_DIR / "attention_variants.csv"
