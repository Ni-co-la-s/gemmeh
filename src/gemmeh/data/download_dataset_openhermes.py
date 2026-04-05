"""Download teknium/openhermes from Hugging Face and save it as local JSON.
This dataset contains only 1 input/output, no multi turn conversation.

Usage:
    uv run -m gemmeh.data.download_dataset_openhermes
"""

import json
import os

from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = "data/openhermes_raw"
OUTPUT_PATH = f"{OUTPUT_DIR}/openhermes.json"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset: teknium/openhermes (split=train, streaming=True)")
    dataset = load_dataset("teknium/openhermes", split="train", streaming=True)

    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")
        for example in tqdm(dataset, desc="Writing openhermes"):
            if count > 0:
                f.write(",\n")
            f.write(json.dumps(example, ensure_ascii=False))
            count += 1
        f.write("\n]\n")

    print(f"Saved {count:,} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
