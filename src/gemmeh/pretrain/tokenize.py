"""
Tokenize JSONL corpus to flat binary files for LLM pretraining.

Reads documents from JSONL, encodes with SentencePiece, appends <eos> after
each document, and writes token IDs as a contiguous uint16 numpy memmap.

The output is a flat array of token IDs with no padding. Documents are
separated by eos tokens. The training data loader reads fixed-length windows
from this array and the eos token indicates boundaries.

Usage:
    uv run -m gemmeh.pretrain.tokenize \
        --model data/tokenizers/run_32k_1B/sentencepiece.model \
        --train_input data/fineweb_raw/finewebedu.jsonl \
        --val_input data/fineweb_raw/finewebedu_val.jsonl \
        --train_output data/tokenized/train.bin \
        --val_output data/tokenized/val.bin \
        --workers 8
"""

import argparse
import json
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import sentencepiece as spm


def tokenize_chunk(args: tuple) -> list[int]:
    """Tokenize a chunk of JSONL lines. Runs in a worker process."""
    model_path, lines = args

    # Each worker loads its own SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    eos_id = sp.eos_id()

    all_ids = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "").strip()
        except json.JSONDecodeError:
            continue
        if not text:
            continue

        ids = sp.encode(text, out_type=int)
        ids.append(eos_id)
        all_ids.extend(ids)

    return all_ids


def tokenize_file(
    model_path: str,
    input_path: str,
    output_path: str,
    workers: int,
    chunk_size: int = 5_000,
):
    """Tokenize a JSONL file to a flat uint16 binary file."""
    import time

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"  Tokenizing {input_path} -> {output_path}")
    print(f"  Workers: {workers}, chunk size: {chunk_size}")

    # First pass: count lines for progress tracking
    print("  Counting lines...")
    t0 = time.time()
    total_lines = 0
    with open(input_path, "rb") as f:
        while True:
            buf = f.read(1024 * 1024 * 64)  # 64MB chunks
            if not buf:
                break
            total_lines += buf.count(b"\n")
    print(f"  Total lines: {total_lines:,} (counted in {time.time() - t0:.1f}s)")

    # Second pass: tokenize in parallel, streaming results to disk
    temp_path = output_path + ".tmp"
    total_tokens = 0
    lines_processed = 0
    t0 = time.time()

    def chunk_generator(f_in):
        """Yield (model_path, chunk_of_lines) tuples for workers."""
        chunk = []
        for line in f_in:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield (model_path, chunk)
                chunk = []
        if chunk:
            yield (model_path, chunk)

    with open(input_path, "r", encoding="utf-8") as f_in, open(temp_path, "wb") as f_out:
        pool = Pool(workers)

        # imap_unordered streams results back as they complete,
        for token_ids in pool.imap_unordered(tokenize_chunk, chunk_generator(f_in)):
            if token_ids:
                arr = np.array(token_ids, dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += len(token_ids)

            lines_processed += chunk_size
            elapsed = time.time() - t0
            rate = lines_processed / elapsed if elapsed > 0 else 0
            eta = (total_lines - lines_processed) / rate if rate > 0 else 0
            print(
                f"    ~{lines_processed:,}/{total_lines:,} "
                f"({min(100, lines_processed / total_lines * 100):.1f}%) "
                f"| {total_tokens / 1e9:.3f}B tokens "
                f"| {elapsed:.0f}s elapsed "
                f"| ETA {eta:.0f}s",
                end="\r",
            )

        pool.close()
        pool.join()

    # Rename temp to final
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)

    # Verify memory-mapping
    mmap = np.memmap(output_path, dtype=np.uint16, mode="r")
    assert len(mmap) == total_tokens, f"Mismatch: file has {len(mmap)} tokens, expected {total_tokens}"

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"\n  Done: {total_tokens:,} tokens ({total_tokens / 1e9:.2f}B), {size_gb:.2f} GB on disk")

    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize JSONL corpus to binary memmap files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to sentencepiece.model")
    parser.add_argument("--train_input", required=True, help="Training JSONL file")
    parser.add_argument("--val_input", required=True, help="Validation JSONL file")
    parser.add_argument("--train_output", required=True, help="Output path for train.bin")
    parser.add_argument("--val_output", required=True, help="Output path for val.bin")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="Number of worker processes")

    args = parser.parse_args()

    # Verify tokenizer loads
    sp = spm.SentencePieceProcessor()
    sp.load(args.model)
    print(f"Tokenizer: {args.model} (vocab: {sp.get_piece_size()}, eos_id: {sp.eos_id()})")

    # Check vocab fits in uint16
    assert sp.get_piece_size() <= 65535, f"Vocab size {sp.get_piece_size()} exceeds uint16 max (65535)"

    print("\n[1/2] Training data")
    train_tokens = tokenize_file(args.model, args.train_input, args.train_output, args.workers)

    print("\n[2/2] Validation data")
    val_tokens = tokenize_file(args.model, args.val_input, args.val_output, args.workers)

    print("\nSummary:")
    print(f"  Train: {train_tokens / 1e9:.2f}B tokens ({os.path.getsize(args.train_output) / 1e9:.2f} GB)")
    print(f"  Val:   {val_tokens / 1e9:.2f}B tokens ({os.path.getsize(args.val_output) / 1e9:.2f} GB)")
    print(f"  Total: {(train_tokens + val_tokens) / 1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
