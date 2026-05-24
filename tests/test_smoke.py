"""Phase 1 smoke tests: CUDA availability, model load, forward-pass shape."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import config
from src.model_loader import load_model


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for this project"


def test_model_loads():
    tokenizer, model = load_model(config.DEFAULT_MODEL)
    assert tokenizer is not None
    assert model is not None
    assert next(model.parameters()).device.type == "cuda"
    assert next(model.parameters()).dtype == torch.float16


def test_forward_pass_logits_shape():
    tokenizer, model = load_model(config.DEFAULT_MODEL)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(config.DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs)
    vocab_size = model.config.vocab_size
    assert outputs.logits.shape == (1, seq_len, vocab_size)
