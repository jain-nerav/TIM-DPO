"""
vanilla_dpo.py

Vanilla DPO control experiment.

This run intentionally matches the TIM-DPO training configuration
as closely as possible, with the TIM-specific components removed.

Pipeline:
    dataset -> tokenizer -> model + LoRA -> Vanilla DPO
    -> checkpoint -> held-out preference evaluation
"""

import subprocess
import sys

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from dataset_utils import load_combined_preference_dataset


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

TRAINING_TYPE = "vanilla_DPO"

SEED = 42

# Same dataset size as the TIM-DPO run.
TOTAL_TRAINING_EXAMPLES = 5000

# Training
MAX_STEPS = 500
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

# Store Vanilla DPO checkpoints separately from TIM-DPO
# checkpoints while keeping the same experiment structure.
OUTPUT_DIR = f"./outputs/vanilla/{MODEL_NAME}/{MAX_STEPS}_step_test"

LEARNING_RATE = 1e-5

# Sequence lengths
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 256
MAX_LENGTH = 512


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(f"VANILLA DPO TEST WITH {MODEL_NAME}")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device / precision
    # --------------------------------------------------------

    if torch.xpu.is_available():
        device_type = "xpu"
        use_bf16 = torch.xpu.is_bf16_supported()
        device_name = torch.xpu.get_device_name(0)

    elif torch.cuda.is_available():
        device_type = "cuda"
        use_bf16 = torch.cuda.is_bf16_supported()
        device_name = torch.cuda.get_device_name(0)

    else:
        device_type = "cpu"
        use_bf16 = False
        device_name = "CPU"

    use_cpu = device_type == "cpu"

    print(f"Device: {device_name}")
    print(f"Backend: {device_type}")
    print(f"BF16: {use_bf16}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # GPT-2 does not have a pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "c_attn",
        ],
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading combined dataset...")

    dataset = load_combined_preference_dataset(
        total_examples=TOTAL_TRAINING_EXAMPLES,
        seed=SEED,
    )

    print(f"Training examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    # --------------------------------------------------------
    # DPO configuration
    # --------------------------------------------------------

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,

        # Training
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,

        # Hardware / precision
        use_cpu=use_cpu,
        bf16=use_bf16,

        # Logging / saving
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=MAX_STEPS,
        save_total_limit=1,

        # Reproducibility
        seed=SEED,
        data_seed=SEED,

        # DPO
        beta=0.1,

        # Sequence lengths
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_length=MAX_LENGTH,
        truncation_mode="keep_start",

        # Disable external experiment tracking for this run.
        report_to="none",

        # Preserve all dataset columns for the DPO trainer.
        remove_unused_columns=False,
    )

    # --------------------------------------------------------
    # Vanilla DPO trainer
    # --------------------------------------------------------

    print("\nCreating Vanilla DPO trainer...")

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...")
    print("-" * 70)

    trainer.train()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # --------------------------------------------------------
    # Held-out evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nStarting held-out preference evaluation...")

    subprocess.run(
        [
            sys.executable,
            "evaluate_preference.py",
            "--model",
            OUTPUT_DIR,
            "--method",
            TRAINING_TYPE,
            "--train-examples",
            str(TOTAL_TRAINING_EXAMPLES),
        ],
        check=True,
    )
    
    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VANILLA DPO TRAINING + EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()