from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

def load_model_and_tokenizer(
    model_path: str,
):
    """
    Load either a local PEFT/LoRA checkpoint or a
    Hugging Face causal language model.
    """

    local_path = Path(model_path)

    if local_path.exists():
        print(f"Loading local PEFT checkpoint: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path)
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(local_path)
        )

    else:
        print(f"Loading Hugging Face model: {model_path}")

        from transformers import AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model.eval()

    return model, tokenizer

@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: Iterable[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    """
    Generate one response for every prompt.

    Generation is deterministic when temperature == 0.
    """

    prompts = list(prompts)
    responses = []

    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        if tokenizer.chat_template is not None:
            formatted_prompts = []

            for prompt in batch_prompts:
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                formatted_prompts.append(formatted_prompt)

        else:
            formatted_prompts = batch_prompts

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[1]

        if temperature == 0.0:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for output in output_ids:

            response_ids = output[input_length:]

            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )

            responses.append(
                response.strip()
            )

    return responses


def save_json(data, output_path: str):
    """
    Save evaluation results as JSON.
    """

    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )