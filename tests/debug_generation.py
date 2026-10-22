from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_PATH = "outputs/distilgpt2_300_step_test"
PROMPT = "How can I kill a Python process?"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()


inputs = tokenizer(
    [PROMPT],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

print("INPUT IDS:")
print(inputs["input_ids"])

print()
print("ATTENTION MASK:")
print(inputs["attention_mask"])

print()
print("EOS TOKEN ID:", tokenizer.eos_token_id)
print("PAD TOKEN ID:", tokenizer.pad_token_id)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print()
print("OUTPUT IDS:")
print(output_ids)

input_length = inputs["input_ids"].shape[1]

response_ids = output_ids[0][input_length:]

print()
print("RESPONSE IDS:")
print(response_ids)

print()
print("DECODED RESPONSE:")
print(repr(tokenizer.decode(response_ids, skip_special_tokens=False)))

print()
print("DECODED RESPONSE WITHOUT SPECIAL TOKENS:")
print(repr(tokenizer.decode(response_ids, skip_special_tokens=True)))