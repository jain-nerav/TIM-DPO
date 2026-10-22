"""
main.py

TIM-DPO

Main training entry point.

This script runs the end-to-end TIM-DPO training pipeline:
    dataset -> tokenizer -> model + LoRA -> TIM-DPO -> checkpoint
    -> held-out preference evaluation
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from peft import LoraConfig
from trl import DPOConfig

from config import TIMConfig
from dataset_utils import load_combined_preference_dataset
from weighted_dpo import WeightedDPOTrainer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

TRAINING_TYPE = "TIM_DPO"

SEED = 42

# Total number of training examples used for this run.
# Must be >10 to prevent ZeroDivisonError
TOTAL_TRAINING_EXAMPLES = 10

# Training
MAX_STEPS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

OUTPUT_DIR = f"./outputs/tim_dpo/{MODEL_NAME}/{MAX_STEPS}_step_test"

LEARNING_RATE = 1e-5

# Sequence lengths
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 256
MAX_LENGTH = 512


# ============================================================
# Training
# ============================================================

def run_training(
    tim_config: TIMConfig,
    output_dir: str,
    training_type: str = TRAINING_TYPE,
):
    """
    Run one complete TIM-DPO training experiment.

    Args:
        tim_config:
            Token Importance Memory configuration for this run.

        output_dir:
            Directory where the trained checkpoint and run
            configuration are saved.

        training_type:
            Label passed to the held-out preference evaluator.
    """

    print("=" * 70)
    print(f"TIM-DPO TRAINING WITH {MODEL_NAME}")
    print("=" * 70)

    output_dir = str(Path(output_dir))

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
    # TIM configuration
    # --------------------------------------------------------

    print("\nTIM configuration:")
    print(tim_config)

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
        output_dir=output_dir,

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

        # Preserve all dataset columns for the custom DPO trainer.
        remove_unused_columns=False,
    )

    # --------------------------------------------------------
    # TIM-DPO trainer
    # --------------------------------------------------------

    print("\nCreating TIM-DPO trainer...")

    trainer = WeightedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        tim_config=tim_config,
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

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the resolved TIM configuration alongside the checkpoint.
    config_path = Path(output_dir) / "tim_config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            tim_config.__dict__,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # TIM statistics
    # --------------------------------------------------------

    print("\nTIM statistics:")
    print(trainer.tim.statistics())

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
            output_dir,
            "--method",
            training_type,
            "--train-examples",
            str(TOTAL_TRAINING_EXAMPLES),
        ],
        check=True,
    )

    print("\n" + "=" * 70)
    print("TIM-DPO TRAINING + EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    tim_config = TIMConfig()

    run_training(
        tim_config=tim_config,
        output_dir=OUTPUT_DIR,
        training_type=TRAINING_TYPE,
    )


if __name__ == "__main__":
    main()