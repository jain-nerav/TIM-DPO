from pathlib import Path

import torch
from safetensors.torch import load_file


MODELS = {
    "Vanilla DPO": Path(
        "outputs/distilgpt2_vanilla_500_step_test"
    ),
    "TIM-DPO": Path(
        "outputs/distilgpt2_500_TIM_DPO_step_test"
    ),
}


def inspect_checkpoint(name, path):
    print()
    print("=" * 80)
    print(f"{name}")
    print(f"Path: {path}")
    print("=" * 80)

    adapter_file = path / "adapter_model.safetensors"

    if not adapter_file.exists():
        print(f"ERROR: Could not find {adapter_file}")
        return

    state_dict = load_file(adapter_file)

    print(f"Adapter tensors: {len(state_dict)}")

    total_parameters = 0
    total_squared = 0.0
    max_abs = 0.0

    print()
    print("Adapter parameters:")
    print("-" * 80)

    for name, tensor in state_dict.items():
        tensor = tensor.float()

        num_parameters = tensor.numel()
        norm = torch.linalg.vector_norm(tensor).item()
        max_value = tensor.abs().max().item()

        total_parameters += num_parameters
        total_squared += tensor.pow(2).sum().item()
        max_abs = max(max_abs, max_value)

        print(
            f"{name:<60} "
            f"shape={str(tuple(tensor.shape)):<20} "
            f"norm={norm:.8f}"
        )

    total_norm = total_squared ** 0.5

    print()
    print("-" * 80)
    print(f"Total adapter parameters: {total_parameters:,}")
    print(f"Total adapter L2 norm:    {total_norm:.8f}")
    print(f"Maximum absolute value:   {max_abs:.8f}")


def main():
    print("=" * 80)
    print("LORA ADAPTER INSPECTION")
    print("=" * 80)

    for name, path in MODELS.items():
        inspect_checkpoint(name, path)


if __name__ == "__main__":
    main()