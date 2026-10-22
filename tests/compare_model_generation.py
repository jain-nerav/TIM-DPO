from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


PROMPTS = [
    "Explain why the sky is blue.",
    "What is 2 + 2?",
    "Write a short paragraph about computers.",
    "How does a Python loop work?",
    "What is the capital of France?",
]

BASE_MODEL = "distilgpt2"

MODELS = {
    "Base": BASE_MODEL,
    "Vanilla DPO": "outputs/distilgpt2_vanilla_500_step_test",
    "TIM-DPO": "outputs/distilgpt2_500_TIM_DP0_step_test",
}

MAX_NEW_TOKENS = 64


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if Path(model_path).exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()

    if torch.xpu.is_available():
        model = model.to("xpu")
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        model = model.to("cuda")
        device = torch.device("cuda")
    else:
        model = model.to("cpu")
        device = torch.device("cpu")

    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]

    response_ids = output_ids[0][input_length:]

    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )

    return response


def main():
    print("=" * 70)
    print("CONTROLLED GENERATION COMPARISON")
    print("=" * 70)

    for model_name, model_path in MODELS.items():
        print()
        print("=" * 70)
        print(model_name)
        print(f"Model: {model_path}")
        print("=" * 70)

        model, tokenizer, device = load_model(model_path)

        print(f"Device: {device}")

        for i, prompt in enumerate(PROMPTS, start=1):
            response = generate(
                model,
                tokenizer,
                device,
                prompt,
            )

            print()
            print(f"[{i}] Prompt:")
            print(prompt)
            print()
            print("Response:")
            print(repr(response))

        del model

        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()