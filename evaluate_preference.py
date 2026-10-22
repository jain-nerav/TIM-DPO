"""
evaluate_preference.py

TIM-DPO

Final held-out preference evaluation.

This evaluator measures a trained preference-optimization model
on a fixed held-out evaluation split that is disjoint from the
training examples.

Metrics:
    - Preference accuracy
    - Mean chosen reward
    - Mean rejected reward
    - Mean reward margin

Rewards are reference-relative:

    r(x, y) = beta * (
        log pi_policy(y | x)
        - log pi_reference(y | x)
    )

Preference accuracy is the fraction of examples for which the
chosen response receives a larger reference-relative reward
than the rejected response.

The evaluator also reports metrics separately for each source dataset.

This script is intended for final experiments rather than smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import AutoPeftModelForCausalLM

from dataset_utils import load_evaluation_dataset

# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 8

BETA = 0.1

MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 256
MAX_LENGTH = 512

TRUNCATION_MODE = "keep_start"

DEBUG = False

# ============================================================
# Tokenization
# ============================================================

def prepare_example(tokenizer, example):
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    # --------------------------------------------------
    # Tokenize the prompt and responses separately.
    # No special tokens, matching the trainer.
    # --------------------------------------------------

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    chosen_ids = tokenizer(
        chosen,
        truncation=True,
        max_length=MAX_COMPLETION_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    rejected_ids = tokenizer(
        rejected,
        truncation=True,
        max_length=MAX_COMPLETION_LENGTH,
        add_special_tokens=False,
    )["input_ids"]

    # --------------------------------------------------
    # Build prompt + response sequences.
    # --------------------------------------------------

    chosen_full = prompt_ids + chosen_ids
    rejected_full = prompt_ids + rejected_ids

    # --------------------------------------------------
    # Match the trainer's maximum sequence length.
    #
    # keep_start:
    # retain the beginning of the prompt + response sequence.
    # --------------------------------------------------
    if len(chosen_full) > MAX_LENGTH:
        chosen_full = chosen_full[:MAX_LENGTH]

    if len(rejected_full) > MAX_LENGTH:
        rejected_full = rejected_full[:MAX_LENGTH]

    return {
        "chosen_ids": chosen_full,
        "rejected_ids": rejected_full,
        "prompt_length_chosen": min(
            len(prompt_ids),
            len(chosen_full),
        ),
        "prompt_length_rejected": min(
            len(prompt_ids),
            len(rejected_full),
        ),
        "source": example["source"],
    }


def collate_batch(tokenizer, examples):
    """
    Pad chosen and rejected sequences independently so that
    each can be evaluated in its own batch tensor.
    """
    chosen_ids = [
        example["chosen_ids"]
        for example in examples
    ]

    rejected_ids = [
        example["rejected_ids"]
        for example in examples
    ]

    chosen_lengths = [
        example["prompt_length_chosen"]
        for example in examples
    ]

    rejected_lengths = [
        example["prompt_length_rejected"]
        for example in examples
    ]

    sources = [
        example["source"]
        for example in examples
    ]

    chosen_batch = tokenizer.pad(
        {
            "input_ids": chosen_ids,
        },
        padding=True,
        return_tensors="pt",
    )

    rejected_batch = tokenizer.pad(
        {
            "input_ids": rejected_ids,
        },
        padding=True,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_batch["input_ids"],
        "chosen_attention_mask": chosen_batch["attention_mask"],
        "rejected_input_ids": rejected_batch["input_ids"],
        "rejected_attention_mask": rejected_batch["attention_mask"],
        "prompt_length_chosen": torch.tensor(
            chosen_lengths,
            dtype=torch.long,
        ),
        "prompt_length_rejected": torch.tensor(
            rejected_lengths,
            dtype=torch.long,
        ),
        "sources": sources,
    }


# ============================================================
# Log-probability calculation
# ============================================================

@torch.no_grad()
def logprobs_for_response(
    model,
    input_ids,
    attention_mask,
    prompt_length,
):
    """
    Calculate the summed log probability of the response tokens only.

    The prompt contributes context to the model's predictions but
    does not contribute to the returned sequence score.
    """

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # Use double precision when accumulating token log probabilities
    # so that small policy/reference differences are preserved.
    token_log_probs = token_log_probs.double()

    sequence_lengths = attention_mask.sum(dim=1)

    response_log_probs = []

    for i in range(input_ids.size(0)):

        start = max(
            prompt_length[i].item() - 1,
            0,
        )

        end = max(
            sequence_lengths[i].item() - 1,
            start,
        )

        response_log_probs.append(
            token_log_probs[i, start:end].sum()
        )

    return torch.stack(response_log_probs)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    policy_model,
    reference_model,
    tokenizer,
    evaluation_dataset,
    device,
):
    """
    Evaluate the trained policy against the fixed reference model
    on the full held-out preference dataset.

    For each chosen/rejected pair, compute reference-relative
    rewards and determine whether the chosen response receives
    the larger reward.
    """

    policy_model.eval()
    reference_model.eval()
    print(
        f"Policy model device:    "
        f"{next(policy_model.parameters()).device}"
    )

    print(
        f"Reference model device: "
        f"{next(reference_model.parameters()).device}"
    )

    examples = []

    for example in evaluation_dataset:
        examples.append(
            prepare_example(
                tokenizer,
                example,
            )
        )

    loader = DataLoader(
        examples,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            tokenizer,
            batch,
        ),
    )

    total_examples = 0
    total_correct = 0

    all_chosen_rewards = []
    all_rejected_rewards = []
    all_margins = []

    source_metrics = {}

    first_batch = True

    for batch in loader:

        chosen_input_ids = batch[
            "chosen_input_ids"
        ].to(device)

        chosen_attention_mask = batch[
            "chosen_attention_mask"
        ].to(device)

        rejected_input_ids = batch[
            "rejected_input_ids"
        ].to(device)

        rejected_attention_mask = batch[
            "rejected_attention_mask"
        ].to(device)

        chosen_prompt_lengths = batch[
            "prompt_length_chosen"
        ].to(device)

        rejected_prompt_lengths = batch[
            "prompt_length_rejected"
        ].to(device)

        # ----------------------------------------------------
        # Policy model
        # ----------------------------------------------------

        policy_chosen_logps = logprobs_for_response(
            policy_model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_prompt_lengths,
        )

        policy_rejected_logps = logprobs_for_response(
            policy_model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_prompt_lengths,
        )

        # ----------------------------------------------------
        # Reference model
        # ----------------------------------------------------

        reference_chosen_logps = logprobs_for_response(
            reference_model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_prompt_lengths,
        )

        reference_rejected_logps = logprobs_for_response(
            reference_model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_prompt_lengths,
        )

        # ----------------------------------------------------
        # Reference-relative rewards.
        #
        # Each response reward measures how much the policy's
        # log probability differs from the reference model's
        # log probability, scaled by the DPO beta coefficient.
        # ----------------------------------------------------

        chosen_rewards = BETA * (
            policy_chosen_logps
            - reference_chosen_logps
        )

        rejected_rewards = BETA * (
            policy_rejected_logps
            - reference_rejected_logps
        )

        margins = (
            chosen_rewards
            - rejected_rewards
        )

        # A preference is counted as correct when the chosen response
        # has a larger reference-relative reward than the rejected response.
        correct = margins > 0

        # ----------------------------------------------------
        # Overall aggregation
        # ----------------------------------------------------

        total_examples += len(batch["sources"])
        total_correct += int(
            correct.sum().item()
        )

        all_chosen_rewards.extend(
            chosen_rewards.cpu().tolist()
        )

        all_rejected_rewards.extend(
            rejected_rewards.cpu().tolist()
        )

        all_margins.extend(
            margins.cpu().tolist()
        )

        # ----------------------------------------------------
        # Per-source aggregation
        # ----------------------------------------------------

        for i, source in enumerate(
            batch["sources"]
        ):

            if source not in source_metrics:

                source_metrics[source] = {
                    "examples": 0,
                    "correct": 0,
                    "chosen_rewards": [],
                    "rejected_rewards": [],
                    "margins": [],
                }

            metrics = source_metrics[source]

            metrics["examples"] += 1

            metrics["correct"] += int(
                correct[i].item()
            )

            metrics["chosen_rewards"].append(
                chosen_rewards[i].item()
            )

            metrics["rejected_rewards"].append(
                rejected_rewards[i].item()
            )

            metrics["margins"].append(
                margins[i].item()
            )

        # ----------------------------------------------------
        # Optional first-batch diagnostic.
        #
        # Useful for checking policy/reference log-probability
        # differences before running the full evaluation.
        # ----------------------------------------------------

        if DEBUG and first_batch:

            print("\nFIRST BATCH DIAGNOSTIC")
            print("-" * 70)

            print(
                f"Policy chosen logprob:      "
                f"{policy_chosen_logps[0].item():.12e}"
            )

            print(
                f"Reference chosen logprob:   "
                f"{reference_chosen_logps[0].item():.12e}"
            )

            print(
                f"Policy rejected logprob:    "
                f"{policy_rejected_logps[0].item():.12e}"
            )

            print(
                f"Reference rejected logprob: "
                f"{reference_rejected_logps[0].item():.12e}"
            )

            print(
                f"Chosen logprob difference:  "
                f"{(policy_chosen_logps[0] - reference_chosen_logps[0]).item():.12e}"
            )

            print(
                f"Rejected logprob difference:"
                f" {(policy_rejected_logps[0] - reference_rejected_logps[0]).item():.12e}"
            )

            first_batch = False

    # ========================================================
    # Final overall metrics
    # ========================================================

    mean_chosen_reward = (
        sum(all_chosen_rewards)
        / len(all_chosen_rewards)
    )

    mean_rejected_reward = (
        sum(all_rejected_rewards)
        / len(all_rejected_rewards)
    )

    mean_reward_margin = (
        sum(all_margins)
        / len(all_margins)
    )

    preference_accuracy = (
        total_correct
        / total_examples
    )

    # ========================================================
    # Final per-source metrics
    # ========================================================

    by_source = {}

    for source, metrics in source_metrics.items():

        examples_count = metrics["examples"]

        by_source[source] = {
            "examples": examples_count,
            "preference_accuracy": (
                metrics["correct"]
                / examples_count
            ),
            "mean_chosen_reward": (
                sum(metrics["chosen_rewards"])
                / examples_count
            ),
            "mean_rejected_reward": (
                sum(metrics["rejected_rewards"])
                / examples_count
            ),
            "mean_reward_margin": (
                sum(metrics["margins"])
                / examples_count
            ),
        }

    return {
        "examples": total_examples,
        "preference_accuracy": preference_accuracy,
        "mean_chosen_reward": mean_chosen_reward,
        "mean_rejected_reward": mean_rejected_reward,
        "mean_reward_margin": mean_reward_margin,
        "by_source": by_source,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained policy checkpoint.",
    )

    parser.add_argument(
        "--method",
        required=True,
        help="Training method used to produce the checkpoint.",
    )

    parser.add_argument(
        "--train-examples",
        type=int,
        required=True,
        help="Number of training examples used to construct the held-out split.",
    )

    parser.add_argument(
        "--eval-examples",
        type=int,
        default=None,
        help="Number of held-out evaluation examples.",
    )

    args = parser.parse_args()

    CHECKPOINT_PATH = str(
        Path(args.model).resolve()
    )

    MODEL_NAME = "distilgpt2"

    TRAIN_EXAMPLES = args.train_examples

    if args.eval_examples is None:
        EVAL_EXAMPLES = round(
            TRAIN_EXAMPLES / 10
        )
    else:
        EVAL_EXAMPLES = args.eval_examples

    project_root = Path(__file__).resolve().parent
    outputs_root = project_root / "outputs"
    results_root = project_root / "results"

    checkpoint_path = Path(CHECKPOINT_PATH)

    relative_checkpoint_dir = checkpoint_path.relative_to(
        outputs_root
    )

    OUTPUT_PATH = (
        results_root
        / relative_checkpoint_dir
        / "preference_eval.json"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.xpu.is_available():
        device = torch.device("xpu")
        device_name = torch.xpu.get_device_name(0)

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)

    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"\nDevice:  {device}")
    print(f"GPU:     {device_name}")


    print(f"Policy checkpoint: {CHECKPOINT_PATH}")
    print(f"Reference model:   {MODEL_NAME}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    print("Loading trained policy...")

    policy_model = AutoPeftModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH
    )

    policy_model.to(device)
    policy_model.eval()

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    print("Loading reference model...")

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    reference_model.to(device)
    reference_model.eval()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading held-out evaluation dataset...")

    evaluation_dataset = load_evaluation_dataset(
        train_examples=TRAIN_EXAMPLES,
        eval_examples=EVAL_EXAMPLES,
        seed=SEED,
    )


    print(
        f"Evaluation examples: "
        f"{len(evaluation_dataset)}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nRunning evaluation...")
    print("-" * 70)

    results = evaluate_model(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        evaluation_dataset=evaluation_dataset,
        device=device,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\nOverall results:")

    print(
        f"  Examples:              "
        f"{results['examples']}"
    )

    print(
        f"  Preference accuracy:   "
        f"{results['preference_accuracy']:.4f}"
    )

    print(
        f"  Mean chosen reward:    "
        f"{results['mean_chosen_reward']:.12e}"
    )

    print(
        f"  Mean rejected reward:  "
        f"{results['mean_rejected_reward']:.12e}"
    )

    print(
        f"  Mean reward margin:    "
        f"{results['mean_reward_margin']:.12e}"
    )

    print("\nResults by dataset:")

    for source, metrics in results[
        "by_source"
    ].items():

        print(f"\n  {source}:")

        print(
            f"    Examples:             "
            f"{metrics['examples']}"
        )

        print(
            f"    Preference accuracy:  "
            f"{metrics['preference_accuracy']:.4f}"
        )

        print(
            f"    Mean chosen reward:   "
            f"{metrics['mean_chosen_reward']:.12e}"
        )

        print(
            f"    Mean rejected reward: "
            f"{metrics['mean_rejected_reward']:.12e}"
        )

        print(
            f"    Mean reward margin:   "
            f"{metrics['mean_reward_margin']:.12e}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = OUTPUT_PATH

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "method": args.method,
        "seed": SEED,
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT_PATH,
        "beta": BETA,
        "batch_size": BATCH_SIZE,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "max_length": MAX_LENGTH,
        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()