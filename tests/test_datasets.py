"""
test_datasets.py

Sanity check for the TIM-DPO dataset pipeline.
"""

from dataset_utils import load_combined_preference_dataset
from dataset_utils import (
    load_ultrafeedback,
    load_anthropic_hh,
    load_saferlhf,
)


def print_example(name, dataset):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    example = dataset[0]

    print("\nPROMPT:")
    print(example["prompt"])

    print("\nCHOSEN:")
    print(example["chosen"])

    print("\nREJECTED:")
    print(example["rejected"])

    print("\nTypes:")
    print("prompt:  ", type(example["prompt"]))
    print("chosen:  ", type(example["chosen"]))
    print("rejected:", type(example["rejected"]))


def main():

    ultrafeedback = load_ultrafeedback(
        split="train_prefs"
    )

    anthropic_hh = load_anthropic_hh(
        split="train"
    )

    saferlhf = load_saferlhf(
        split="train"
    )

    print_example(
        "UltraFeedback",
        ultrafeedback
    )

    print_example(
        "Anthropic HH-RLHF",
        anthropic_hh
    )

    print_example(
        "SafeRLHF",
        saferlhf
    )

    print("\n" + "=" * 70)
    print("COMBINED DATASET")
    print("=" * 70)

    combined = load_combined_preference_dataset(
        total_examples=300,
        seed=42,
    )

    print("Total examples:", len(combined))
    print("Columns:", combined.column_names)

    print("\nFirst combined example:")
    print(combined[0])


if __name__ == "__main__":
    main()

