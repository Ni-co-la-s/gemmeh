"""
Download FineWeb-Edu dataset with recency-weighted sampling.
Saves as JSONL with metadata for clean document boundaries.
Automatically splits out a validation set at the end.

Output format (one JSON object per line):
{
    "text": "...",
    "id": "...",
    "url": "...",
    "dump": "CC-MAIN-2023-50",
    "language": "en",
    "language_score": 0.95,
    "token_count": 342,
    "score": 3.8,
    "int_score": 4
}

Documents are separated by newlines in the JSONL file.

Usage:
uv run -m gemmeh.data.download_dataset_finewebedu.py
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import TypedDict

OUTPUT_DIR = "data/fineweb_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# After downloading tokens_per_dump for train, continue streaming
# tokens_per_dump * VAL_PROPORTION more tokens into the val file.
VAL_PROPORTION = 0.0005

# Recency-weighted: 2023 gets the most tokens, older years get fewer.
# Targets are in tokens (using the token_count field from the dataset).


class DumpGroup(TypedDict):
    dumps: list[str]
    target_tokens: int


DUMP_SCHEDULE: dict[str, DumpGroup] = {
    "2023": {
        "dumps": [
            "CC-MAIN-2023-50",
            "CC-MAIN-2023-40",
            "CC-MAIN-2023-23",
            "CC-MAIN-2023-14",
            "CC-MAIN-2023-06",
        ],
        "target_tokens": 8_000_0,  # 8B tokens
    },
    "2022": {
        "dumps": [
            "CC-MAIN-2022-49",
            "CC-MAIN-2022-40",
            "CC-MAIN-2022-33",
            "CC-MAIN-2022-27",
            "CC-MAIN-2022-21",
            "CC-MAIN-2022-05",
        ],
        "target_tokens": 5_000_0,  # 5B tokens
    },
    "2021": {
        "dumps": [
            "CC-MAIN-2021-49",
            "CC-MAIN-2021-43",
            "CC-MAIN-2021-39",
            "CC-MAIN-2021-31",
            "CC-MAIN-2021-25",
            "CC-MAIN-2021-21",
            "CC-MAIN-2021-17",
            "CC-MAIN-2021-10",
            "CC-MAIN-2021-04",
        ],
        "target_tokens": 3_000_0,  # 3B tokens
    },
    "2018-2020": {
        "dumps": [
            "CC-MAIN-2020-40",
            "CC-MAIN-2020-16",
            "CC-MAIN-2019-35",
            "CC-MAIN-2019-13",
            "CC-MAIN-2018-39",
            "CC-MAIN-2018-17",
        ],
        "target_tokens": 2_000_0,  # 2B tokens
    },
    "2013-2017": {
        "dumps": [
            "CC-MAIN-2017-39",
            "CC-MAIN-2017-13",
            "CC-MAIN-2016-36",
            "CC-MAIN-2016-07",
            "CC-MAIN-2015-32",
            "CC-MAIN-2015-06",
            "CC-MAIN-2014-23",
            "CC-MAIN-2013-48",
        ],
        "target_tokens": 2_000_0,  # 2B tokens
    },
}

FIELDS_TO_KEEP = ["id", "url", "language", "language_score", "token_count", "score", "int_score"]

output_path = os.path.join(OUTPUT_DIR, "finewebedu.jsonl")
val_path = os.path.join(OUTPUT_DIR, "finewebedu_val.jsonl")
total_tokens = 0
total_val_tokens = 0
total_docs = 0
total_val_docs = 0

with open(output_path, "w", encoding="utf-8") as f_out, open(val_path, "w", encoding="utf-8") as f_val:
    for group_name, group in DUMP_SCHEDULE.items():
        dumps = group["dumps"]
        group_target = group["target_tokens"]
        tokens_per_dump = group_target // len(dumps)
        val_tokens_per_dump = int(tokens_per_dump * VAL_PROPORTION)
        group_tokens = 0
        group_val_tokens = 0
        group_docs = 0
        group_val_docs = 0

        print(f"\n{'=' * 60}")
        print(
            f"  {group_name}: {group_target / 1e9:.0f}B tokens "
            f"across {len(dumps)} dumps ({tokens_per_dump / 1e9:.2f}B each)"
        )
        print(f"{'=' * 60}")

        for dump in dumps:
            dump_tokens = 0
            dump_val_tokens = 0
            dump_docs = 0
            dump_val_docs = 0
            print(f"\n  Loading {dump}...")

            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name=dump,
                    split="train",
                    streaming=True,
                )
            except Exception as e:
                print(f"  ⚠ Skipping {dump}: {e}")
                continue

            pbar = tqdm(
                total=tokens_per_dump + val_tokens_per_dump,
                desc=f"  {dump}",
                unit="tok",
                unit_scale=True,
                unit_divisor=1000,
            )

            writing_val = False

            for example in ds:
                text = example.get("text", "").strip()
                if not text:
                    continue

                tok_count = example.get("token_count", 0)
                if tok_count == 0:
                    # Fallback estimate if field is missing
                    tok_count = len(text) // 4

                # Build record with metadata
                record = {"text": text, "dump": dump}
                for field in FIELDS_TO_KEEP:
                    if field in example and field != "token_count":
                        record[field] = example[field]
                record["token_count"] = tok_count

                line = json.dumps(record, ensure_ascii=False) + "\n"

                if not writing_val:
                    f_out.write(line)
                    dump_tokens += tok_count
                    dump_docs += 1
                    pbar.update(tok_count)
                    if dump_tokens >= tokens_per_dump:
                        writing_val = True
                else:
                    f_val.write(line)
                    dump_val_tokens += tok_count
                    dump_val_docs += 1
                    pbar.update(tok_count)
                    if dump_val_tokens >= val_tokens_per_dump:
                        break

            pbar.close()

            group_tokens += dump_tokens
            group_val_tokens += dump_val_tokens
            group_docs += dump_docs
            group_val_docs += dump_val_docs
            print(
                f"    {dump}: {dump_tokens / 1e9:.2f}B train ({dump_docs:,} docs) "
                f"+ {dump_val_tokens / 1e6:.1f}M val ({dump_val_docs:,} docs)"
            )

        total_tokens += group_tokens
        total_val_tokens += group_val_tokens
        total_docs += group_docs
        total_val_docs += group_val_docs
        print(f"\n  {group_name} total: {group_tokens / 1e9:.2f}B train, {group_val_tokens / 1e6:.1f}M val")

print(f"\n{'=' * 60}")
print("  COMPLETE")
print(f"  Train: {total_tokens / 1e9:.1f}B tokens, {total_docs:,} docs → {output_path}")
print(f"  Val:   {total_val_tokens / 1e6:.1f}M tokens, {total_val_docs:,} docs → {val_path}")
print(f"{'=' * 60}")
