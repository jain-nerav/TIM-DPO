import torch
from transformers import TrainerCallback
from datasets import Dataset
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import DPOConfig

from weighted_dpo import WeightedDPOTrainer


MODEL_NAME = "distilbert/distilgpt2"

from config import TIMConfig

config = TIMConfig()


# --------------------------------------------------
# Model and tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --------------------------------------------------
# UltraFeedback Binarized
# --------------------------------------------------

SEED = 42
NUM_EXAMPLES = 200


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs",
)

dataset = dataset.shuffle(seed=SEED)
dataset = dataset.select(range(NUM_EXAMPLES))


# --------------------------------------------------
# Convert conversations to plain text
# --------------------------------------------------

def messages_to_text(messages):

    if isinstance(messages, str):
        return messages

    return "\n".join(
        message["content"]
        for message in messages
        if message["role"] != "system"
    )

def convert_to_text(example):
    prompt = messages_to_text(example["prompt"])

    chosen = example["chosen"][-1]["content"]
    rejected = example["rejected"][-1]["content"]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


dataset = dataset.map(convert_to_text)

dataset = dataset.select_columns([
    "prompt",
    "chosen",
    "rejected",
])

MAX_LENGTH = 1000

def fits_context(example):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False
    )["input_ids"]

    chosen_ids = tokenizer(
        example["chosen"],
        add_special_tokens=False
    )["input_ids"]

    rejected_ids = tokenizer(
        example["rejected"],
        add_special_tokens=False
    )["input_ids"]

    return (
        len(prompt_ids) + len(chosen_ids) <= MAX_LENGTH
        and
        len(prompt_ids) + len(rejected_ids) <= MAX_LENGTH
    )

before = len(dataset)
dataset = dataset.filter(fits_context)

print(
    f"Filtered {before - len(dataset)} examples "
    f"exceeding {MAX_LENGTH} tokens."
)
print("Remaining:", len(dataset))

dataset = dataset.select(range(min(200, len(dataset))))


# --------------------------------------------------
# Dataset check
# --------------------------------------------------

print("\n===== DATASET CHECK =====")

print("Number of examples:", len(dataset))
print("Columns:", dataset.column_names)

for i in range(3):

    print(f"\n===== Example {i} =====")

    print("\nPROMPT:")
    print(dataset[i]["prompt"])

    print("\nCHOSEN:")
    print(dataset[i]["chosen"])

    print("\nREJECTED:")
    print(dataset[i]["rejected"])

    print("\nCharacter lengths:")
    print("Chosen:", len(dataset[i]["chosen"]))
    print("Rejected:", len(dataset[i]["rejected"]))
    
# --------------------------------------------------
# DPO configuration
# --------------------------------------------------

training_args = DPOConfig(
    output_dir="./debug_output",
    per_device_train_batch_size=1,
    max_steps=1000,
    learning_rate=1e-5,
    logging_steps=10,
    report_to=[],
    bf16=False,
    fp16=False,
)

# --------------------------------------------------
# Weighted DPO trainer
# --------------------------------------------------

trainer = WeightedDPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,    
    train_dataset=dataset,
    tim_config=config,
)

class TIMLoggingCallback(TrainerCallback):
    def __init__(self, trainer_getter, token_ids, log_every=10):
        self.trainer_getter = trainer_getter
        self.token_ids = token_ids
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return control

        trainer = self.trainer_getter()

        print(f"\n===== TIM SNAPSHOT | STEP {state.global_step} =====")

        for token_id in self.token_ids:
            raw = trainer.tim.ema_table[token_id].item()
            token_text = tokenizer.decode([token_id])

            print(
                f"ID={token_id:5d} | "
                f"token={token_text!r:15s} | "
                f"TIM={raw:.6f}"
            )

        return control

# trainer.add_callback(
#     TIMLoggingCallback(
#         trainer_getter=lambda: trainer,
#         token_ids=tracked_token_ids,
#         log_every=10,
#     )
# )

# --------------------------------------------------
# Run one training step
# --------------------------------------------------

trainer.train()

# --------------------------------------------------
# Final TIM summary
# --------------------------------------------------

updated = trainer.tim.update_counts > 0
tim_values = trainer.tim.ema_table[updated]

print("\n===== FINAL TIM SUMMARY =====")
print("Updated tokens:", updated.sum().item())
print("TIM mean:", tim_values.mean().item())
print("TIM min:", tim_values.min().item())
print("TIM max:", tim_values.max().item())