"""Load a tokenizer + model in fp16 onto CUDA with eager attention.

Eager attention is used for every phase so per-decode-step behavior stays
explicit and comparable across phases (CLAUDE.md hardware constraints).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def load_model(model_name: str = config.DEFAULT_MODEL):
    """Return (tokenizer, model) for ``model_name``.

    Model is loaded in float16, placed on CUDA, set to eval mode, and forced
    to use the eager attention implementation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.to(config.DEVICE)
    model.eval()
    return tokenizer, model
