from __future__ import annotations

import argparse
from pathlib import Path



# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "benchmarks"
    / "alpacaeval"
)

DEFAULT_MAX_NEW_TOKENS = 1024


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_alpacaeval_dataset():
    """
    Load the official AlpacaEval evaluation set from
    the local JSON file.

    The file contains the standard 805-example
    AlpacaEval evaluation set.
    """

    import json

    dataset_path = (
        Path(__file__).resolve().parent
        / "data"
        / "alpaca_eval.json"
    )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if len(dataset) != 805:
        raise ValueError(
            f"Expected 805 AlpacaEval examples, "
            f"found {len(dataset)}."
        )

    required_fields = {
        "instruction",
        "output",
    }

    for i, example in enumerate(dataset):
        if not required_fields.issubset(example):
            raise ValueError(
                f"AlpacaEval example {i} is missing "
                f"required fields."
            )

    return dataset


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate_alpacaeval_outputs(
    model_path: str,
    output_path: str,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    limit: int | None = None,
):
    """
    Generate responses for the official AlpacaEval evaluation set.

    The resulting JSON is directly compatible with the official
    AlpacaEval evaluator.
    """
    from benchmarks.common import (
        generate_responses,
        load_model_and_tokenizer,
    )


    print("=" * 70)
    print("ALPACAEVAL GENERATION")
    print("=" * 70)

    print(f"Model: {model_path}")

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    dataset = load_alpacaeval_dataset()

    if limit is not None:
        dataset = dataset[:min(limit, len(dataset))]

    print(f"Prompts: {len(dataset)}")

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    model, tokenizer = load_model_and_tokenizer(
        model_path
    )

    # -------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------

    prompts = [
        example["instruction"]
        for example in dataset
    ]

    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # -------------------------------------------------------------
    # Build official AlpacaEval format
    # -------------------------------------------------------------

    outputs = []

    for example, response in zip(
        dataset,
        responses,
    ):
        outputs.append(
            {
                "instruction": example["instruction"],
                "output": response,
                "generator": Path(model_path).name,
            }
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            outputs,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("GENERATION COMPLETE")
    print("-" * 70)
    print(f"Examples: {len(outputs)}")
    print(f"Saved to: {output_path}")

    return outputs


# ---------------------------------------------------------------------
# Official AlpacaEval evaluation
# ---------------------------------------------------------------------

def evaluate_alpacaeval(
    model_outputs_path: str,
    output_dir: str,
):
    """
    Evaluate saved model outputs using the official AlpacaEval package.

    AlpacaEval handles:
        - GPT-4 Turbo reference outputs
        - GPT-4 Turbo judging
        - preference annotations
        - length-controlled win rate
        - leaderboard/annotation output
    """

    import alpaca_eval

    model_outputs_path = Path(model_outputs_path)
    output_dir = Path(output_dir)

    if not model_outputs_path.exists():
        raise FileNotFoundError(
            f"Model outputs do not exist: {model_outputs_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("ALPACAEVAL 2.0 EVALUATION")
    print("=" * 70)

    print(f"Model outputs: {model_outputs_path}")
    print(f"Output directory: {output_dir}")
    print(
        "Annotator: weighted_alpaca_eval_gpt4_turbo"
    )
    print()

    reference_outputs_path = (
        Path(__file__).resolve().parent
        / "data"
        / "alpaca_eval.json"
    )

    leaderboard = alpaca_eval.evaluate(
        model_outputs=str(model_outputs_path),
        reference_outputs=str(reference_outputs_path),
        annotators_config="weighted_alpaca_eval_gpt4_turbo",
        output_path=str(output_dir),
    )

    print()
    print("=" * 70)
    print("ALPACAEVAL EVALUATION COMPLETE")
    print("=" * 70)

    print(leaderboard)

    return leaderboard

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate AlpacaEval 2.0 outputs."
    )

    parser.add_argument(
        "--model",
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--output",
        help="Path to the AlpacaEval model outputs JSON.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate an existing model outputs JSON.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for AlpacaEval evaluation results.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts generated at once.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens generated per prompt.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of evaluation examples.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Evaluation-only mode
    # -------------------------------------------------------------

    if args.evaluate:
        if args.output is None:
            parser.error(
                "--output is required with --evaluate."
            )

        evaluate_alpacaeval(
            model_outputs_path=args.output,
            output_dir=args.output_dir,
        )

        return

    # -------------------------------------------------------------
    # Generation mode
    # -------------------------------------------------------------

    if args.model is None:
        parser.error(
            "--model is required for generation."
        )

    if args.output is None:
        parser.error(
            "--output is required for generation."
        )

    generate_alpacaeval_outputs(
        model_path=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()