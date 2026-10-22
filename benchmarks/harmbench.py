from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from benchmarks.common import load_model_and_tokenizer, generate_responses


def load_harmbench_behaviors(
    behaviors_path: str,
    limit: Optional[int] = None,
):
    """
    Load HarmBench text behaviors from the official behavior CSV.
    """

    behaviors = []

    with open(behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"BehaviorID", "Behavior"}

        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"HarmBench CSV must contain columns {required}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            behaviors.append(
                {
                    "behavior_id": row["BehaviorID"],
                    "behavior": row["Behavior"],
                }
            )

            if limit is not None and len(behaviors) >= limit:
                break

    if not behaviors:
        raise ValueError("No HarmBench behaviors were loaded.")

    return behaviors


def run_harmbench_generation(
    model_path: str,
    behaviors_path: str,
    output_path: str,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
):
    """
    Generate model completions for HarmBench behaviors.

    This function performs generation only.
    Official HarmBench classification and ASR
    evaluation are performed separately.
    """

    behaviors = load_harmbench_behaviors(
        behaviors_path=behaviors_path,
        limit=limit,
    )

    model, tokenizer = load_model_and_tokenizer(model_path)

    prompts = [
        item["behavior"]
        for item in behaviors
    ]

    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    results = []

    for item, response in zip(behaviors, responses):
        results.append(
            {
                "behavior_id": item["behavior_id"],
                "behavior": item["behavior"],
                "response": response,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "harmbench",
                "num_behaviors": len(results),
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"HarmBench behaviors evaluated: "
        f"{len(results)}"
    )
    print(
        f"Saved completions to: "
        f"{output_path}"
    )

    return results