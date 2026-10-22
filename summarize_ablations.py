"""
summarize_ablations.py

TIM-DPO

Collect preference-evaluation results from all ablation
experiments into a single CSV summary.

Each row represents one ablation configuration.
"""

import csv
import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ABLATION_RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "ablations"
)

SUMMARY_PATH = (
    ABLATION_RESULTS_ROOT / "summary.csv"
)


# ============================================================
# Helpers
# ============================================================

def find_result_files():
    """Find all preference-evaluation result files."""

    return sorted(
        ABLATION_RESULTS_ROOT.glob(
            "*/*/preference_eval.json"
        )
    )


def load_result(result_path):
    """Load one preference-evaluation result."""

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def extract_summary_row(
    result_path: Path,
    result: dict,
):
    """Extract one summary row from a result file."""

    ablation_name = result_path.parent.parent.name
    variant_name = result_path.parent.name

    evaluation_results = result.get(
        "results",
        {},
    )

    by_source = evaluation_results.get(
        "by_source",
        {},
    )

    ultrafeedback = by_source.get(
        "ultrafeedback",
        {},
    )

    anthropic = by_source.get(
        "anthropic",
        {},
    )

    saferlhf = by_source.get(
        "saferlhf",
        {},
    )

    return {
        "ablation": ablation_name,
        "variant": variant_name,
        "method": result.get("method"),
        "model": result.get("model"),
        "checkpoint": result.get("checkpoint"),
        "beta": result.get("beta"),
        "batch_size": result.get("batch_size"),
        "max_prompt_length": result.get(
            "max_prompt_length"
        ),
        "max_completion_length": result.get(
            "max_completion_length"
        ),
        "max_length": result.get("max_length"),
        "examples": evaluation_results.get(
            "examples"
        ),
        "preference_accuracy": evaluation_results.get(
            "preference_accuracy"
        ),
        "mean_chosen_reward": evaluation_results.get(
            "mean_chosen_reward"
        ),
        "mean_rejected_reward": evaluation_results.get(
            "mean_rejected_reward"
        ),
        "mean_reward_margin": evaluation_results.get(
            "mean_reward_margin"
        ),
        "ultrafeedback_examples": ultrafeedback.get(
            "examples",
            0,
        ),
        "anthropic_examples": anthropic.get(
            "examples",
            0,
        ),
        "saferlhf_examples": saferlhf.get(
            "examples",
            0,
        ),
        "result_path": str(result_path),
    }


def write_summary(rows):
    """Write all summary rows to CSV."""

    fieldnames = [
        "ablation",
        "variant",
        "method",
        "model",
        "checkpoint",
        "beta",
        "batch_size",
        "max_prompt_length",
        "max_completion_length",
        "max_length",
        "examples",
        "preference_accuracy",
        "mean_chosen_reward",
        "mean_rejected_reward",
        "mean_reward_margin",
        "ultrafeedback_examples",
        "anthropic_examples",
        "saferlhf_examples",
        "result_path",
    ]

    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    result_files = find_result_files()

    if not result_files:
        raise FileNotFoundError(
            "No preference_eval.json files were found "
            f"under {ABLATION_RESULTS_ROOT}"
        )

    print("=" * 70)
    print("TIM-DPO ABLATION SUMMARY")
    print("=" * 70)

    print(
        f"\nFound {len(result_files)} "
        "ablation result files."
    )

    rows = []

    for result_path in result_files:

        print(
            f"Reading: "
            f"{result_path.relative_to(PROJECT_ROOT)}"
        )

        result = load_result(result_path)

        rows.append(
            extract_summary_row(
                result_path,
                result,
            )
        )

    write_summary(rows)

    print("\n" + "=" * 70)
    print("SUMMARY COMPLETE")
    print("=" * 70)

    print(f"\nRows written: {len(rows)}")
    print(f"Output:       {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
