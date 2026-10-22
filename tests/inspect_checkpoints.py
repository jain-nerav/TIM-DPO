from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODELS = {
    "Base": "distilgpt2",
    "Vanilla DPO": "outputs/distilgpt2_vanilla_500_step_test",
    "TIM-DPO": "outputs/distilgpt2_500_TIM_DPO_step_test",
}

PROMPTS = [
    "Explain why the sky is blue.",
    "What is 2 + 2?",
    "What is the capital of France?",
]


def get_device():
    if torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def inspect_model(model_name, model_path, device):
    print()
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print(f"PATH:  {model_path}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    for prompt in PROMPTS:
        print()
        print("-" * 80)
        print(f"Prompt: {prompt}")
        print("-" * 80)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[:, -1, :]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        top_probs, top_ids = torch.topk(
            probabilities,
            k=10,
            dim=-1,
        )

        print()
        print("Top 10 next-token predictions:")
        print("-" * 80)

        for rank, (token_id, probability) in enumerate(
            zip(top_ids[0], top_probs[0]),
            start=1,
        ):
            token_id = token_id.item()
            probability = probability.item()
            token = tokenizer.decode([token_id])

            print(
                f"{rank:2d}. "
                f"ID={token_id:<6} "
                f"prob={probability:.6f} "
                f"token={repr(token)}"
            )

        newline_ids = tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )

        print()
        print(f"Newline token IDs: {newline_ids}")

        for newline_id in newline_ids:
            newline_probability = probabilities[
                0, newline_id
            ].item()

            print(
                f"Newline ID {newline_id} probability: "
                f"{newline_probability:.6f}"
            )


def main():
    print("=" * 80)
    print("DPO CHECKPOINT NEXT-TOKEN PROBABILITY COMPARISON")
    print("=" * 80)

    device = get_device()
    print(f"Device: {device}")

    for model_name, model_path in MODELS.items():
        inspect_model(
            model_name,
            model_path,
            device,
        )


if __name__ == "__main__":
    main()