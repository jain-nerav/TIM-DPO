from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_NAME = "gpt2"

PROMPTS = [
    "Explain why the sky is blue.",
    "What is 2 + 2?",
    "What is the capital of France?",
]


def main():
    print("=" * 70)
    print("NEXT-TOKEN PROBABILITY DIAGNOSTIC")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print(f"Device: {device}")

    for prompt in PROMPTS:
        print()
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print("=" * 70)

        # GPT-2 is a causal completion model, not a chat model.
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

        # Logits for the next token after the prompt.
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
        print("-" * 70)

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


if __name__ == "__main__":
    main()