"""
test_dataset_split.py

TIM-DPO

Sanity test for the training/evaluation dataset split.
"""

from dataset_utils import load_train_eval_datasets


def main():

    print("=" * 70)
    print("TIM-DPO DATASET SPLIT TEST")
    print("=" * 70)

    train_dataset, eval_dataset = load_train_eval_datasets(
        train_examples=30,
        eval_examples=12,
        seed=42,
    )

    # --------------------------------------------------------
    # Basic sizes
    # --------------------------------------------------------

    print(f"\nTraining examples:   {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    assert len(train_dataset) == 30
    assert len(eval_dataset) == 12

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    expected_columns = {
        "prompt",
        "chosen",
        "rejected",
        "source",
    }

    assert set(train_dataset.column_names) == expected_columns
    assert set(eval_dataset.column_names) == expected_columns

    print("\nPASS: Required columns are present.")

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    train_sources = {}
    eval_sources = {}

    for source in train_dataset["source"]:
        train_sources[source] = train_sources.get(source, 0) + 1

    for source in eval_dataset["source"]:
        eval_sources[source] = eval_sources.get(source, 0) + 1

    print("\nTraining source distribution:")
    for source, count in sorted(train_sources.items()):
        print(f"  {source}: {count}")

    print("\nEvaluation source distribution:")
    for source, count in sorted(eval_sources.items()):
        print(f"  {source}: {count}")

    assert train_sources == {
        "anthropic_hh": 10,
        "saferlhf": 10,
        "ultrafeedback": 10,
    }

    assert eval_sources == {
        "anthropic_hh": 4,
        "saferlhf": 4,
        "ultrafeedback": 4,
    }

    print("\nPASS: Dataset sources are balanced.")

    # --------------------------------------------------------
    # Check for overlap
    # --------------------------------------------------------

    train_examples = {
        (
            example["prompt"],
            example["chosen"],
            example["rejected"],
        )
        for example in train_dataset
    }

    eval_examples = {
        (
            example["prompt"],
            example["chosen"],
            example["rejected"],
        )
        for example in eval_dataset
    }

    overlap = train_examples.intersection(
        eval_examples
    )

    print(f"\nTrain/evaluation overlap: {len(overlap)}")

    assert len(overlap) == 0

    print("PASS: Training and evaluation sets are disjoint.")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SPLIT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
