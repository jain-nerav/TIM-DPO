from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_XSTEST_DATASET = (
    PROJECT_ROOT
    / "benchmarks"
    / "data"
    / "xstest_prompts.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xstest"
)

DEFAULT_HARMBENCH_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "harmbench"
)

DEFAULT_ALPACAEVAL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "alpacaeval"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=False,
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--benchmark",
        required=True,
        choices=[
            "xstest",
            "harmbench",
            "alpacaeval",
        ],
        help="Benchmark to run.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts processed in each generation batch.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens generated per prompt.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark examples to process.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation stage for AlpacaEval instead of generation.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to an existing AlpacaEval model outputs JSON file.",
    )

    parser.add_argument(
        "--behaviors-path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "HarmBench"
            / "data"
            / "behavior_datasets"
            / "harmbench_behaviors_text_test.csv"
        ),
    )

    args = parser.parse_args()

    if not args.evaluate and args.model is None:
        parser.error(
            "--model is required for benchmark generation."
        )

    if args.benchmark == "xstest":
        from benchmarks.xstest import evaluate_xstest

        model_name = Path(args.model).name

        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"{model_name}_results.json"
        )

        evaluate_xstest(
            model_path=args.model,
            xstest_csv=str(DEFAULT_XSTEST_DATASET),
            output_path=str(output_path),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )

    elif args.benchmark == "harmbench":
        from benchmarks.harmbench import run_harmbench_generation
        model_name = Path(args.model).name

        output_path = (
            DEFAULT_HARMBENCH_OUTPUT_DIR
            / f"{model_name}_completions.json"
        )

        run_harmbench_generation(
            model_path=args.model,
            behaviors_path=args.behaviors_path,
            output_path=str(output_path),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )

    elif args.benchmark == "alpacaeval":
        
        from benchmarks.alpacaeval import (
            evaluate_alpacaeval,
            generate_alpacaeval_outputs,
        )

        # -------------------------------------------------------------
        # Official AlpacaEval evaluation
        #
        # Uses the generated model outputs as input to AlpacaEval's
        # official evaluation and judging pipeline.
        # -------------------------------------------------------------
        if args.evaluate:

            if args.output is None:
                parser.error(
                    "--output is required when using "
                    "--evaluate with AlpacaEval."
                )

            evaluate_alpacaeval(
                model_outputs_path=args.output,
                output_dir=str(
                    DEFAULT_ALPACAEVAL_OUTPUT_DIR
                ),
            )

        # -------------------------------------------------------------
        # AlpacaEval model generation
        #
        # Generate outputs in the format expected by AlpacaEval's
        # evaluation pipeline.
        # -------------------------------------------------------------
        else:

            model_name = Path(args.model).name

            output_path = (
                DEFAULT_ALPACAEVAL_OUTPUT_DIR
                / f"{model_name}_outputs.json"
            )

            generate_alpacaeval_outputs(
                model_path=args.model,
                output_path=str(output_path),
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                limit=args.limit,
            )

if __name__ == "__main__":
    main()