from pathlib import Path

import torch
from safetensors.torch import load_file


VANILLA_PATH = Path(
    "outputs/distilgpt2_vanilla_500_step_test/adapter_model.safetensors"
)

TIM_PATH = Path(
    "outputs/distilgpt2_500_TIM_DPO_step_test/adapter_model.safetensors"
)


def main():
    print("=" * 80)
    print("VANILLA DPO vs TIM-DPO ADAPTER DIFFERENCE")
    print("=" * 80)

    vanilla = load_file(VANILLA_PATH)
    tim = load_file(TIM_PATH)

    total_diff_squared = 0.0
    total_vanilla_squared = 0.0

    for name in vanilla:
        vanilla_tensor = vanilla[name].float()
        tim_tensor = tim[name].float()

        difference = tim_tensor - vanilla_tensor

        diff_norm = torch.linalg.vector_norm(
            difference
        ).item()

        vanilla_norm = torch.linalg.vector_norm(
            vanilla_tensor
        ).item()

        max_diff = difference.abs().max().item()

        total_diff_squared += difference.pow(2).sum().item()
        total_vanilla_squared += vanilla_tensor.pow(2).sum().item()

        relative_difference = (
            diff_norm / vanilla_norm
            if vanilla_norm > 0
            else 0.0
        )

        print()
        print(name)
        print("-" * 80)
        print(f"Vanilla norm:        {vanilla_norm:.10f}")
        print(f"TIM-DPO norm:        {torch.linalg.vector_norm(tim_tensor).item():.10f}")
        print(f"Difference norm:     {diff_norm:.10f}")
        print(f"Relative difference: {relative_difference:.10f}")
        print(f"Maximum difference:  {max_diff:.10f}")

    total_difference_norm = total_diff_squared ** 0.5
    total_vanilla_norm = total_vanilla_squared ** 0.5

    print()
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)

    print(
        f"Vanilla adapter norm:     "
        f"{total_vanilla_norm:.10f}"
    )

    print(
        f"TIM-DPO vs Vanilla diff:  "
        f"{total_difference_norm:.10f}"
    )

    print(
        f"Relative difference:       "
        f"{total_difference_norm / total_vanilla_norm:.10f}"
    )


if __name__ == "__main__":
    main()