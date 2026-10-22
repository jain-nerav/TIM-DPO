"""
run_ablation.py

TIM-DPO

Run one-factor-at-a-time TIM-DPO ablation experiments.

Each ablation changes exactly one TIM configuration parameter
from the default TIM-DPO configuration.
"""

import argparse
from pathlib import Path

from config import TIMConfig
from main import run_training


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ABLATION_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "ablations"
)


ABLATIONS = {
    "gradient_metric": {
        "l2": {
            "importance_metric": "l2",
        },
        "l2_squared": {
            "importance_metric": "l2_squared",
        },
    },

    "ema_decay": {
        "beta_0.90": {
            "beta": 0.90,
        },
        "beta_0.95": {
            "beta": 0.95,
        },
        "beta_0.99": {
            "beta": 0.99,
        },
        "beta_0.995": {
            "beta": 0.995,
        },
    },

    "clipping": {
        "disabled": {
            "clip_importance": False,
        },
        "enabled": {
            "clip_importance": True,
        },
    },

    "compression": {
        "none": {
            "compression": "none",
        },
        "log": {
            "compression": "log",
        },
    },

    "batch_normalization": {
        "disabled": {
            "normalize_batch_importance": False,
        },
        "enabled": {
            "normalize_batch_importance": True,
        },
    },

    "memory_mode": {
        "ema": {
            "memory_mode": "ema",
        },
        "previous_batch": {
            "memory_mode": "previous_batch",
        },
    },
}


# ============================================================
# Helpers
# ============================================================

def list_ablations():
    """Print all available ablation configurations."""

    print("\nAvailable TIM-DPO ablations:\n")

    for ablation_name, variants in ABLATIONS.items():

        print(ablation_name)

        for variant_name in variants:
            print(f"  - {variant_name}")

        print()


def build_tim_config(
    ablation_name: str,
    variant_name: str,
):
    """
    Construct a TIMConfig for one ablation.

    The configuration starts from TIMConfig defaults and
    modifies only the parameter belonging to the selected
    ablation.
    """

    try:
        changes = ABLATIONS[
            ablation_name
        ][variant_name]

    except KeyError:
        raise ValueError(
            f"Unknown ablation: "
            f"{ablation_name}/{variant_name}"
        )

    return TIMConfig(**changes)


def build_output_path(
    ablation_name: str,
    variant_name: str,
):
    """Build the output directory for one ablation run."""

    return (
        ABLATION_OUTPUT_ROOT
        / ablation_name
        / variant_name
    )


def run_single_ablation(
    ablation_name: str,
    variant_name: str,
):
    """
    Run one TIM-DPO ablation experiment.
    """

    tim_config = build_tim_config(
        ablation_name,
        variant_name,
    )

    output_dir = build_output_path(
        ablation_name,
        variant_name,
    )

    # --------------------------------------------------------
    # Display experiment
    # --------------------------------------------------------

    print("=" * 70)
    print("TIM-DPO ABLATION")
    print("=" * 70)

    print(f"\nAblation: {ablation_name}")
    print(f"Value:    {variant_name}")
    print(f"Output:   {output_dir}")

    print("\nTIM configuration:")
    print(tim_config)

    print("\nStarting ablation run...\n")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    run_training(
        tim_config=tim_config,
        output_dir=str(output_dir),
        training_type="TIM_DPO",
    )


def run_all_ablations():
    """
    Run every configured TIM-DPO ablation sequentially.
    """

    total_runs = sum(
        len(variants)
        for variants in ABLATIONS.values()
    )

    print("=" * 70)
    print("TIM-DPO FULL ABLATION SUITE")
    print("=" * 70)

    print(f"\nTotal ablation runs: {total_runs}")
    print("Runs will execute sequentially.\n")

    run_number = 0

    for ablation_name, variants in ABLATIONS.items():

        for variant_name in variants:

            run_number += 1

            print("\n" + "=" * 70)
            print(
                f"ABLATION {run_number}/{total_runs}"
            )
            print("=" * 70)

            run_single_ablation(
                ablation_name,
                variant_name,
            )

    print("\n" + "=" * 70)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run TIM-DPO ablation experiments."
    )

    parser.add_argument(
        "--ablation",
        choices=list(ABLATIONS.keys()),
        help="Ablation dimension to run.",
    )

    parser.add_argument(
        "--value",
        help="Specific variant within the selected ablation.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available ablation configurations.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available ablation configurations sequentially.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # List configurations
    # --------------------------------------------------------

    if args.list:
        if args.all or args.ablation or args.value:
            parser.error(
                "--list cannot be combined with "
                "--all, --ablation, or --value."
            )

        list_ablations()
        return

    # --------------------------------------------------------
    # Run all configurations
    # --------------------------------------------------------

    if args.all:

        if args.ablation is not None or args.value is not None:
            parser.error(
                "--all cannot be combined with "
                "--ablation or --value."
            )

        run_all_ablations()
        return

    # --------------------------------------------------------
    # Validate single-ablation arguments
    # --------------------------------------------------------

    if args.ablation is None or args.value is None:
        parser.error(
            "--ablation and --value are required "
            "unless --list or --all is used."
        )

    if args.value not in ABLATIONS[args.ablation]:
        available = ", ".join(
            ABLATIONS[args.ablation].keys()
        )

        parser.error(
            f"Unknown value '{args.value}' for "
            f"ablation '{args.ablation}'. "
            f"Available values: {available}"
        )

    # --------------------------------------------------------
    # Run single ablation
    # --------------------------------------------------------

    run_single_ablation(
        args.ablation,
        args.value,
    )


if __name__ == "__main__":
    main()
