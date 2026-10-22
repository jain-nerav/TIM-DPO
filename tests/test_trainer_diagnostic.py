import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig

from config import TIMConfig
from weighted_dpo import WeightedDPOTrainer
from dataset_utils import load_combined_preference_dataset


MODEL_NAME = "sshleifer/tiny-gpt2"


def main():

    print("=" * 60)
    print("TIM-DPO TRAINER DIAGNOSTIC")
    print("=" * 60)

    # -------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTokenizer loaded.")
    print("EOS token ID:", tokenizer.eos_token_id)
    print("PAD token ID:", tokenizer.pad_token_id)

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # -------------------------------------------------------
    # LoRA
    # -------------------------------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"],
    )

    # -------------------------------------------------------
    # TIM configuration
    # -------------------------------------------------------

    tim_config = TIMConfig(
        memory_mode="ema",
        beta=0.99,
        importance_metric="l2_squared",
        clip_importance=True,
        clip_max=4.0,
        clip_min=0.25,
        normalize_batch_importance=True,
        compression="none",
        normalization="mean",
    )

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    dataset = load_combined_preference_dataset(
        total_examples=10,
        seed=42,
    )

    print("\nDataset size:", len(dataset))

    # -------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------

    training_args = DPOConfig(
        output_dir="./debug_output/tim_diagnostic",

        max_steps=3,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        learning_rate=1e-5,

        use_cpu=True,

        logging_strategy="steps",
        logging_steps=1,

        save_strategy="no",

        seed=42,
        data_seed=42,

        beta=0.1,

        max_prompt_length=128,
        max_completion_length=128,
        max_length=256,

        report_to="none",

        remove_unused_columns=False,
    )

    # -------------------------------------------------------
    # Trainer
    # -------------------------------------------------------

    trainer = WeightedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        tim_config=tim_config,
    )

    # -------------------------------------------------------
    # Initial TIM state
    # -------------------------------------------------------

    print("\nInitial TIM statistics:")
    print(trainer.tim.statistics())

    initial_table = trainer.tim.ema_table.clone()

    # -------------------------------------------------------
    # Train
    # -------------------------------------------------------

    print("\nStarting diagnostic training...\n")

    trainer.train()

    # -------------------------------------------------------
    # Final TIM state
    # -------------------------------------------------------

    final_table = trainer.tim.ema_table.clone()

    print("\nFinal TIM statistics:")
    print(trainer.tim.statistics())

    # -------------------------------------------------------
    # Diagnostic 1:
    # Did TIM actually change?
    # -------------------------------------------------------

    changed = ~torch.isclose(
        initial_table,
        final_table,
    )

    num_changed = changed.sum().item()

    print("\nDiagnostic 1: TIM updates")
    print("Tokens whose TIM changed:", num_changed)

    assert num_changed > 0, (
        "TIM did not update any tokens."
    )

    print("PASS: TIM updated.")

    # -------------------------------------------------------
    # Diagnostic 2:
    # Were untouched tokens preserved?
    # -------------------------------------------------------

    untouched = ~changed

    untouched_values = final_table[untouched]

    if untouched_values.numel() > 0:

        assert torch.allclose(
            untouched_values,
            torch.ones_like(untouched_values),
        )

        print(
            "PASS: untouched tokens remain at "
            "initial importance."
        )

    # -------------------------------------------------------
    # Diagnostic 3:
    # Are importance values non-uniform?
    # -------------------------------------------------------

    unique_values = torch.unique(final_table[changed])

    print(
        "Number of distinct updated TIM values:",
        unique_values.numel(),
    )

    assert unique_values.numel() > 1, (
        "All updated tokens have the same TIM value."
    )

    print("PASS: TIM values are non-uniform.")

    # -------------------------------------------------------
    # Diagnostic 4:
    # Basic statistics sanity check
    # -------------------------------------------------------

    stats = trainer.tim.statistics()

    assert stats["mean"] > 0.0
    assert stats["min"] >= 0.0
    assert stats["max"] >= stats["min"]

    print("PASS: TIM statistics are valid.")

    print("\n" + "=" * 60)
    print("TRAINER DIAGNOSTIC PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()