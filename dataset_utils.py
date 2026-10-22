"""
dataset_utils.py

TIM-DPO

Utilities for loading and combining preference datasets.

The main TIM-DPO training setup uses an equal contribution
from three preference datasets:

    1. UltraFeedback
    2. Anthropic HH-RLHF
    3. PKU-SafeRLHF

All datasets are converted to the common format:

    prompt
    chosen
    rejected
    source
"""

from datasets import load_dataset, concatenate_datasets


# ============================================================
# Dataset Sources
# ============================================================

ULTRAFEEDBACK = "HuggingFaceH4/ultrafeedback_binarized"
ANTHROPIC_HH = "Anthropic/hh-rlhf"
SAFERLHF = "PKU-Alignment/PKU-SafeRLHF"


# ============================================================
# UltraFeedback
# ============================================================

def load_ultrafeedback(split="train_prefs"):
    """
    Load UltraFeedback and convert each preference example
    to the common prompt/chosen/rejected format.
    """

    dataset = load_dataset(
        ULTRAFEEDBACK,
        split=split,
    )

    def convert(example):
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]

        prompt = example["prompt"]

        chosen = chosen_messages[-1]["content"]
        rejected = rejected_messages[-1]["content"]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "ultrafeedback",
        }

    return dataset.map(
        convert,
        remove_columns=dataset.column_names,
    )


# ============================================================
# Anthropic HH-RLHF
# ============================================================

def load_anthropic_hh(split="train"):
    """
    Load Anthropic HH-RLHF.

    Each chosen/rejected field contains the full conversation.
    The shared conversation history is retained as the prompt,
    while the final assistant response becomes the preference
    response.
    """

    dataset = load_dataset(
        ANTHROPIC_HH,
        split=split,
    )

    def convert(example):
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Find the first point where the chosen and rejected
        # conversations diverge.
        #
        # The shared prefix becomes the prompt, while the
        # remaining portions become the chosen and rejected responses.
        prefix_length = 0

        for c, r in zip(chosen, rejected):
            if c != r:
                break

            prefix_length += 1

        prompt = chosen[:prefix_length]

        chosen_response = chosen[prefix_length:]
        rejected_response = rejected[prefix_length:]

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "source": "anthropic_hh",
        }

    return dataset.map(
        convert,
        remove_columns=dataset.column_names,
    )


# ============================================================
# SafeRLHF
# ============================================================

def load_saferlhf(split="train"):
    """
    Load PKU-SafeRLHF.

    The dataset provides two preference signals:
        better_response_id
        safer_response_id

    For the main DPO mixture, better_response_id is used as
    the chosen/rejected preference signal.

    The safety preference remains available in the original
    dataset but is not used as the primary DPO label here.
    """

    dataset = load_dataset(
        SAFERLHF,
        split=split,
    )

    def convert(example):

        responses = [
            example["response_0"],
            example["response_1"],
        ]

        chosen_id = example["better_response_id"]
        rejected_id = 1 - chosen_id

        return {
            "prompt": example["prompt"],
            "chosen": responses[chosen_id],
            "rejected": responses[rejected_id],
            "source": "saferlhf",
        }

    return dataset.map(
        convert,
        remove_columns=dataset.column_names,
    )


# ============================================================
# Sampling
# ============================================================

def sample_dataset(
    dataset,
    num_examples,
    seed,
):
    """
    Randomly sample a fixed number of examples.

    If the requested number exceeds the available examples,
    the full dataset is returned.
    """

    num_examples = min(
        num_examples,
        len(dataset),
    )

    return dataset.shuffle(
        seed=seed,
    ).select(
        range(num_examples)
    )


# ============================================================
# Train / Evaluation Splitting
# ============================================================

def split_dataset(
    dataset,
    train_examples,
    eval_examples,
    seed,
):
    """
    Create disjoint training and evaluation subsets.

    The dataset is shuffled once using the supplied seed.
    The first portion is used for training and the following
    portion is used for evaluation.

    This guarantees that the two subsets do not overlap.
    """

    total_required = train_examples + eval_examples

    if total_required > len(dataset):
        raise ValueError(
            "Requested training and evaluation examples "
            f"({total_required}) exceed dataset size "
            f"({len(dataset)})."
        )

    shuffled = dataset.shuffle(
        seed=seed,
    )

    train_dataset = shuffled.select(
        range(train_examples)
    )

    eval_dataset = shuffled.select(
        range(
            train_examples,
            train_examples + eval_examples,
        )
    )

    return train_dataset, eval_dataset


# ============================================================
# Balanced Dataset Sizes
# ============================================================

def _balanced_counts(total_examples):
    """
    Split a total number of examples as evenly as possible
    across the three source datasets.
    """

    base_examples = total_examples // 3
    remainder = total_examples % 3

    counts = [
        base_examples,
        base_examples,
        base_examples,
    ]

    if remainder >= 1:
        counts[0] += 1

    if remainder >= 2:
        counts[1] += 1

    return counts


# ============================================================
# Combined Dataset
# ============================================================

def load_combined_preference_dataset(
    total_examples=300_000,
    seed=42,
):
    """
    Load and combine the three preference datasets with
    as-even-as-possible contributions.
    """

    (
        n_ultrafeedback,
        n_anthropic,
        n_saferlhf,
    ) = _balanced_counts(total_examples)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ultrafeedback = load_ultrafeedback()
    anthropic_hh = load_anthropic_hh()
    saferlhf = load_saferlhf()

    # --------------------------------------------------------
    # Balanced sampling
    # --------------------------------------------------------

    ultrafeedback = sample_dataset(
        ultrafeedback,
        n_ultrafeedback,
        seed,
    )

    anthropic_hh = sample_dataset(
        anthropic_hh,
        n_anthropic,
        seed + 1,
    )

    saferlhf = sample_dataset(
        saferlhf,
        n_saferlhf,
        seed + 2,
    )

    # --------------------------------------------------------
    # Combine and shuffle
    # --------------------------------------------------------

    combined = concatenate_datasets(
        [
            ultrafeedback,
            anthropic_hh,
            saferlhf,
        ]
    )

    combined = combined.shuffle(
        seed=seed,
    )

    return combined


# ============================================================
# Train / Evaluation Dataset
# ============================================================

def load_train_eval_datasets(
    train_examples=300_000,
    eval_examples=30_000,
    seed=42,
):
    """
    Load balanced training and held-out evaluation datasets.

    Each source dataset contributes as evenly as possible
    to both splits.

    Training and evaluation examples are guaranteed to be
    disjoint within each source dataset.

    Returns:
        train_dataset
        eval_dataset
    """

    (
        train_ultra,
        train_anthropic,
        train_saferlhf,
    ) = _balanced_counts(train_examples)

    (
        eval_ultra,
        eval_anthropic,
        eval_saferlhf,
    ) = _balanced_counts(eval_examples)

    # --------------------------------------------------------
    # Load source datasets
    # --------------------------------------------------------

    ultrafeedback = load_ultrafeedback()
    anthropic_hh = load_anthropic_hh()
    saferlhf = load_saferlhf()

    # --------------------------------------------------------
    # Create disjoint source-specific splits
    # --------------------------------------------------------

    ultra_train, ultra_eval = split_dataset(
        ultrafeedback,
        train_ultra,
        eval_ultra,
        seed,
    )

    anthropic_train, anthropic_eval = split_dataset(
        anthropic_hh,
        train_anthropic,
        eval_anthropic,
        seed + 1,
    )

    saferlhf_train, saferlhf_eval = split_dataset(
        saferlhf,
        train_saferlhf,
        eval_saferlhf,
        seed + 2,
    )

    # --------------------------------------------------------
    # Combine training split
    # --------------------------------------------------------

    train_dataset = concatenate_datasets(
        [
            ultra_train,
            anthropic_train,
            saferlhf_train,
        ]
    )

    train_dataset = train_dataset.shuffle(
        seed=seed,
    )

    # --------------------------------------------------------
    # Combine evaluation split
    # --------------------------------------------------------

    eval_dataset = concatenate_datasets(
        [
            ultra_eval,
            anthropic_eval,
            saferlhf_eval,
        ]
    )

    eval_dataset = eval_dataset.shuffle(
        seed=seed + 3,
    )

    return train_dataset, eval_dataset

def load_evaluation_dataset(
    train_examples=30_000,
    eval_examples=3_000,
    seed=42,
):
    """
    Load a fixed held-out evaluation dataset.

    The evaluation examples are selected using the same deterministic
    train/evaluation split procedure as load_train_eval_datasets().

    Therefore, using the same train_examples, eval_examples, and
    seed produces the same held-out evaluation examples that would
    be returned by load_train_eval_datasets().

    Each source contributes as evenly as possible to the evaluation set.

    Args:
        train_examples:
            Total number of training examples used to determine
            where the held-out evaluation portion begins.

        eval_examples:
            Total number of evaluation examples across all
            source datasets.

        seed:
            Random seed used for deterministic splitting.

    Returns:
        Balanced held-out evaluation dataset.
    """

    # --------------------------------------------------------
    # Determine per-source split sizes
    # --------------------------------------------------------

    (
        train_ultra,
        train_anthropic,
        train_saferlhf,
    ) = _balanced_counts(train_examples)

    (
        eval_ultra,
        eval_anthropic,
        eval_saferlhf,
    ) = _balanced_counts(eval_examples)

    # --------------------------------------------------------
    # Load source datasets
    # --------------------------------------------------------

    ultrafeedback = load_ultrafeedback()
    anthropic_hh = load_anthropic_hh()
    saferlhf = load_saferlhf()

    # --------------------------------------------------------
    # Create deterministic disjoint splits
    # --------------------------------------------------------

    _, ultra_eval = split_dataset(
        ultrafeedback,
        train_ultra,
        eval_ultra,
        seed,
    )

    _, anthropic_eval = split_dataset(
        anthropic_hh,
        train_anthropic,
        eval_anthropic,
        seed + 1,
    )

    _, saferlhf_eval = split_dataset(
        saferlhf,
        train_saferlhf,
        eval_saferlhf,
        seed + 2,
    )

    # --------------------------------------------------------
    # Combine evaluation split
    # --------------------------------------------------------

    eval_dataset = concatenate_datasets(
        [
            ultra_eval,
            anthropic_eval,
            saferlhf_eval,
        ]
    )

    # Shuffle the combined evaluation set so that examples from
    # different sources are interleaved rather than grouped by source.
    eval_dataset = eval_dataset.shuffle(
        seed=seed + 3,
    )

    return eval_dataset

