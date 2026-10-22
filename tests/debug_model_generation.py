from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "outputs/distilgpt2_500_step_test"

PROMPTS = [
    "Explain why the sky is blue.",
    "What is 2 + 2?",
    "Write a short paragraph about computers.",
    "How does a Python loop work?",
    "What is the capital of France?",
]


print("=" * 70)
print("MODEL GENERATION SANITY CHECK")
print("=" * 70)

print(f"Model: {MODEL_PATH}")
print(f"Prompts: {len(PROMPTS)}")
print()


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.eval()


for index, prompt in enumerate(PROMPTS, start=1):

    print("-" * 70)
    print(f"PROMPT {index}")
    print(f"Input: {prompt}")
    print()

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
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

    print("Response:")
    print(repr(response))
    print()


print("=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)