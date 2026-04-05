"""
Run grid search experiments on the tokenizer creation.
Tests several vocab sizes and 32k vocab size and several vocab sizes on 1B tokens

Usage:
    uv run -m gemmeh.tokenizer.run_experiments
"""

import subprocess
from typing import TypedDict

INPUT = "data/fineweb_raw/finewebedu.jsonl"
VAL = "data/fineweb_raw/finewebedu_val.jsonl"
PROJECT = "gemmeh-2024"


class Experiment(TypedDict):
    output_dir: str
    vocab_size: int
    target_tokens: int


experiments: list[Experiment] = []


# Grid 1: target tokens sweep at 32K vocab
for target_tokens in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
    if target_tokens >= 1_000_000_000:
        label = f"{target_tokens // 1_000_000_000}B"
    else:
        label = f"{target_tokens // 1_000_000}M"
    key = f"data/tokenizers/run_32k_{label}"
    if any(e["output_dir"] == key for e in experiments):
        continue
    experiments.append(
        {
            "output_dir": key,
            "vocab_size": 32768,
            "target_tokens": target_tokens,
        }
    )


# Grid 2: vocab size sweep at 1B tokens
for vocab_size in [8192, 16384, 65536, 131072, 262144]:
    vocab_k = vocab_size // 1024
    experiments.append(
        {
            "output_dir": f"data/tokenizers/run_{vocab_k}k_1B",
            "vocab_size": vocab_size,
            "target_tokens": 1_000_000_000,
        }
    )


print(f"Running {len(experiments)} experiments:\n")
for exp in experiments:
    print(f"  {exp['output_dir']}")
print()

for i, exp in enumerate(experiments):
    print(f"\n{'=' * 60}")
    print(f"  Experiment {i + 1}/{len(experiments)}: {exp['output_dir']}")
    print(f"{'=' * 60}\n")

    cmd: list[str] = [
        "uv",
        "run",
        "-m",
        "gemmeh.tokenizer.pipeline",
        "--input",
        INPUT,
        "--val",
        VAL,
        "--output_dir",
        exp["output_dir"],
        "--vocab_size",
        str(exp["vocab_size"]),
        "--target_tokens",
        str(exp["target_tokens"]),
        "--wandb_project",
        PROJECT,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED: {exp['output_dir']} (exit code {result.returncode})")
    else:
        print(f"\n  DONE: {exp['output_dir']}")

print(f"\n{'=' * 60}")
print(f"  All {len(experiments)} experiments complete.")
print(f"{'=' * 60}")
