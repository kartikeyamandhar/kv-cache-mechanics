"""Phase 1 smoke test: load the default model, generate text, inspect cache.

Run: ``python experiments/phase1_smoke.py``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import config
from src.model_loader import load_model


def main() -> None:
    print(f"Loading {config.DEFAULT_MODEL} on {config.DEVICE} ...")
    tokenizer, model = load_model(config.DEFAULT_MODEL)

    # Device of model parameters.
    param_device = next(model.parameters()).device
    print(f"Model parameter device: {param_device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    prompt = "Explain what a KV cache is in one sentence."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(config.DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=64, do_sample=False
        )
    output_text = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("\n--- Generated text ---")
    print(output_text)
    print("----------------------\n")

    # Confirm past_key_values is produced by a forward pass with use_cache.
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    pkv = outputs.past_key_values
    print(f"past_key_values returned: {pkv is not None}")
    if pkv is not None:
        print(f"past_key_values type: {type(pkv).__name__}")
        try:
            print(f"past_key_values cached length: {pkv.get_seq_length()}")
        except AttributeError:
            print(f"num layers in cache: {len(pkv)}")


if __name__ == "__main__":
    main()
