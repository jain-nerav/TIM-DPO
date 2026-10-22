from __future__ import annotations

import csv

from benchmarks.common import (
    generate_responses,
    load_model_and_tokenizer,
    save_json,
)


XSTEST_URL = (
    "https://raw.githubusercontent.com/"
    "paul-rottger/xstest/main/xstest_prompts.csv"
)


def load_xstest_prompts(csv_path: str):
    """
    Load XSTest prompts from the official CSV.

    Returns:
        A list of dictionaries containing:
            id
            type
            prompt
    """

    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "prompt": row["prompt"],
                }
            )

    return rows


def evaluate_xstest(
    model_path: str,
    xstest_csv: str,
    output_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    limit: int | None = None,
):
    """
    Run XSTest generation and save raw model responses.

    No local classification is performed here.

    The official XSTest classification procedure will be
    applied separately before the final benchmark evaluation.
    """

    print("=" * 70)
    print("XSTEST GENERATION")
    print("=" * 70)

    print(f"Model: {model_path}")
    print(f"Dataset: {xstest_csv}")

    rows = load_xstest_prompts(xstest_csv)

    if limit is not None:
        rows = rows[:limit]

    print(f"Prompts: {len(rows)}")

    model, tokenizer = load_model_and_tokenizer(
        model_path
    )

    prompts = [
        row["prompt"]
        for row in rows
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

    for row, response in zip(rows, responses):
        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "prompt": row["prompt"],
                "response": response,
            }
        )

    output = {
        "benchmark": "XSTest",
        "model": model_path,
        "num_prompts": len(results),
        "results": results,
    }

    save_json(
        output,
        output_path,
    )

    print()
    print("GENERATION COMPLETE")
    print("-" * 70)

    for result in results:
        print()
        print(f"ID: {result['id']}")
        print(f"Type: {result['type']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Response: {result['response']}")

    print()
    print(f"Saved to: {output_path}")

    return output