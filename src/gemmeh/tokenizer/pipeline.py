"""
Train tokenizer with sentencepiece Pipeline: Train, Evaluate, Compare, Log to W&B

Usage:
    uv run gemmeh.tokenizer.pipeline \
        --input data/fineweb_raw/finewebedu.jsonl \
        --val data/fineweb_raw/finewebedu_val.jsonl \
        --output_dir data/tokenizer/run_32k \
        --vocab_size 32768 \
        --wandb_project gemmeh-2024
"""

import argparse
import json
import os
from collections import Counter
from typing import cast
import sentencepiece as spm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from huggingface_hub import snapshot_download

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# Symbols that might be useful for future projects
USER_DEFINED_SYMBOLS = [
    "<start_of_turn>",
    "<end_of_turn>",
    "<start_of_thought>",
    "<end_of_thought>",
    "<|endoftext|>",
]

# Sentences used to do the comparisons in wandb
EXAMPLE_SENTENCES = [
    # Normal prose
    "The president of the United States met with world leaders in Washington.",
    # Scientific
    "The molecule C6H12O6 has a molar mass of 180.16 g/mol.",
    # Hyphen chains
    "The long-term, high-frequency, low-latency trading system malfunctioned mid-session."
    # Financial tickers
    "Shares of BRK.A outperformed S&P 500 ETFs like SPY and VOO in FY2024.",
    # Multilingual
    "El niño comió crème brûlée while discussing über-efficient algorithms.",
    # Code-like content
    "def train(model, lr=3e-4): optimizer = Adam(model.parameters(), lr=lr)",
    # Unicode
    "π ≈ 3.14159 and θ = 45°",
]


# Training
def make_doc_iterator(jsonl_path: str, target_tokens: int | None = None):
    """Create a document iterator for SentencePiece training.

    Returns a generator that yields document texts from a JSONL file.
    If target_tokens is set, stops after approximately that many tokens.
    """
    tokens_seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if not text:
                    continue
                if target_tokens is not None:
                    tok_count = record.get("token_count", len(text) // 4)
                    if tokens_seen >= target_tokens:
                        return
                    tokens_seen += tok_count
                yield text
            except json.JSONDecodeError:
                continue


def train_sentencepiece(input_path: str, output_dir: str, vocab_size: int, target_tokens: int | None = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "sentencepiece")

    # For small runs, disable streaming mode
    is_large = target_tokens is None or target_tokens >= 1_000_000_000

    spm.SentencePieceTrainer.train(
        sentence_iterator=make_doc_iterator(input_path, target_tokens),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        byte_fallback=True,  # Avoid unknown chars
        split_digits=True,  # As done by Gemma3
        split_by_whitespace=True,
        allow_whitespace_only_pieces=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        max_sentence_length=16384,
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=is_large,
    )

    return f"{model_prefix}.model"


# External tokenizers
def ensure_external_tokenizers(model_ids: list[str], external_dir: str) -> dict[str, str]:
    """
    Make sure the tokenizers used for the comparisons actually exist
    """
    os.makedirs(external_dir, exist_ok=True)
    paths = {}
    for model_id in model_ids:
        name = model_id.split("/")[-1]
        local_path = os.path.join(external_dir, name)
        if os.path.exists(os.path.join(local_path, "tokenizer_config.json")):
            paths[name] = local_path
            continue
        if not HAS_HF_HUB:
            continue
        try:
            snapshot_download(
                repo_id=model_id, local_dir=local_path, allow_patterns=["tokenizer.json", "tokenizer_config.json"]
            )
            paths[name] = local_path
        except Exception as e:
            print(f"  Failed to download {model_id}: {e}")
    return paths


def load_tokenizers(our_path: str, external_paths: dict[str, str]) -> dict[str, object]:

    tokenizers = {}
    # Ours: always load as raw SentencePiece
    sp_model = os.path.join(our_path, "sentencepiece.model")
    if os.path.exists(sp_model):
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model)
        tokenizers["ours"] = sp
    # External: load via HuggingFace AutoTokenizer
    for name, path in external_paths.items():
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return tokenizers


# Evaluation helpers (distinguishing between SentencePiece model and Huggingsface tokenizer)
def encode(tok, text: str) -> list[int]:
    if isinstance(tok, spm.SentencePieceProcessor):
        return cast(list[int], tok.encode(text, out_type=int))
    return cast(list[int], tok.encode(text, add_special_tokens=False))


def decode(tok, ids: list[int]) -> str:
    if isinstance(tok, spm.SentencePieceProcessor):
        return cast(str, tok.decode(ids))
    return cast(str, tok.decode(ids, skip_special_tokens=False))


def to_pieces(tok, text: str) -> list[str]:
    if isinstance(tok, spm.SentencePieceProcessor):
        return cast(list[str], tok.encode(text, out_type=str))
    ids = tok.encode(text, add_special_tokens=False)
    return cast(list[str], tok.convert_ids_to_tokens(ids))


def vocab_size(tok) -> int:
    if isinstance(tok, spm.SentencePieceProcessor):
        return cast(int, tok.get_piece_size())
    return cast(int, tok.vocab_size)


def id_to_piece(tok, tid: int) -> str:
    if isinstance(tok, spm.SentencePieceProcessor):
        return cast(str, tok.id_to_piece(tid))
    return cast(str, tok.convert_ids_to_tokens([tid])[0])


def load_val_docs(val_path: str, max_docs: int = 10_000) -> list[str]:
    docs = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "").strip()
                if text:
                    docs.append(text)
                    if len(docs) >= max_docs:
                        break
            except json.JSONDecodeError:
                continue
    return docs


# Evaluation


def evaluate_tokenizer(tok, name: str, val_docs: list[str]) -> dict:
    total_bytes, total_tokens, total_words = 0, 0, 0
    all_ids = []
    tokens_per_doc = []

    for doc in val_docs:
        ids = encode(tok, doc)
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(ids)
        total_words += len(doc.split())
        all_ids.extend(ids)
        tokens_per_doc.append(len(ids))

    freq = Counter(all_ids)
    vs = vocab_size(tok)

    return {
        "name": name,
        "vocab_size": vs,
        "bytes_per_token": total_bytes / total_tokens,
        "tokens_per_word": total_tokens / total_words,
        "vocab_used": len(set(all_ids)),
        "vocab_used_pct": len(set(all_ids)) / vs * 100,
        "avg_tokens_per_doc": total_tokens / len(val_docs),
        "median_tokens_per_doc": sorted(tokens_per_doc)[len(tokens_per_doc) // 2],
        "tokens_per_doc_list": tokens_per_doc,
        "top_50": freq.most_common(50),
        "bottom_50": freq.most_common()[:-51:-1],
    }


def roundtrip_check(tok, val_docs: list[str]) -> dict:
    """
    Validate lossless encode to decode
    """
    failures = 0
    examples: list[dict[str, int | str]] = []
    for i, doc in enumerate(val_docs):
        ids = encode(tok, doc)
        decoded = decode(tok, ids)
        if decoded != doc:
            failures += 1
            if len(examples) < 5:
                for j, (a, b) in enumerate(zip(doc, decoded)):
                    if a != b:
                        examples.append(
                            {
                                "doc_idx": i,
                                "pos": j,
                                "original": doc[max(0, j - 30) : j + 30],
                                "decoded": decoded[max(0, j - 30) : j + 30],
                            }
                        )
                        break
    return {"failures": failures, "failure_rate": failures / len(val_docs) * 100, "examples": examples}


# W&B logging


def log_to_wandb(config, all_metrics, roundtrip, tokenizers, val_docs, output_dir, project):
    # Derive run name from output_dir
    run_name = os.path.basename(output_dir)
    run = wandb.init(project=project, job_type="tokenizer-training", config=config, name=run_name)

    # Scalar metrics per tokenizer
    for name, m in all_metrics.items():
        p = f"tokenizer/{name}"
        run.summary[f"{p}/bytes_per_token"] = m["bytes_per_token"]
        run.summary[f"{p}/tokens_per_word"] = m["tokens_per_word"]
        run.summary[f"{p}/vocab_size"] = m["vocab_size"]
        run.summary[f"{p}/vocab_used_pct"] = m["vocab_used_pct"]
        run.summary[f"{p}/avg_tokens_per_doc"] = m["avg_tokens_per_doc"]

    run.summary["tokenizer/ours/roundtrip_failure_rate"] = roundtrip["failure_rate"]

    # Summary comparison table
    wandb.log(
        {
            "tokenizer/summary": wandb.Table(
                columns=[
                    "tokenizer",
                    "bytes_per_token",
                    "tokens_per_word",
                    "vocab_size",
                    "vocab_used_pct",
                    "avg_tokens_per_doc",
                ],
                data=[
                    [
                        n,
                        round(m["bytes_per_token"], 3),
                        round(m["tokens_per_word"], 3),
                        m["vocab_size"],
                        round(m["vocab_used_pct"], 1),
                        round(m["avg_tokens_per_doc"], 1),
                    ]
                    for n, m in all_metrics.items()
                ],
            )
        }
    )

    # Per-document comparison table
    comp_rows = []
    for i, doc in enumerate(val_docs[:500]):
        row = {"doc_idx": i, "preview": doc[:120], "bytes": len(doc.encode("utf-8"))}
        for name, tok in tokenizers.items():
            row[f"{name}_tokens"] = len(encode(tok, doc))
        comp_rows.append(row)
    if comp_rows:
        cols = list(comp_rows[0].keys())
        wandb.log(
            {
                "tokenizer/doc_comparison": wandb.Table(
                    columns=cols,
                    data=[[r[c] for c in cols] for r in comp_rows],
                )
            }
        )

    # Tokenization examples
    ex_rows = []
    for sentence in EXAMPLE_SENTENCES:
        for name, tok in tokenizers.items():
            pieces = to_pieces(tok, sentence)
            ex_rows.append([sentence, name, " | ".join(pieces), len(pieces)])
    wandb.log(
        {
            "tokenizer/examples": wandb.Table(
                columns=["sentence", "tokenizer", "tokenized", "token_count"],
                data=ex_rows,
            )
        }
    )

    # Top and bottom tokens (ours)
    if "ours" in all_metrics and "ours" in tokenizers:
        tok = tokenizers["ours"]
        for label, key in [("top_tokens", "top_50"), ("bottom_tokens", "bottom_50")]:
            data = [[repr(id_to_piece(tok, tid)), tid, count] for tid, count in all_metrics["ours"][key]]
            wandb.log(
                {
                    f"tokenizer/{label}": wandb.Table(
                        columns=["token", "id", "count"],
                        data=data,
                    )
                }
            )

    # Artifact
    artifact = wandb.Artifact(
        name=f"tokenizer-{config.get('vocab_size', '?')}",
        type="tokenizer",
        metadata=config,
    )
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)
    wandb.finish()


# Main
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Training JSONL (e.g. finewebedu.jsonl)")
    parser.add_argument("--val", required=True, help="Validation JSONL (e.g. finewebedu_val.jsonl)")
    parser.add_argument("--output_dir", default="data/tokenizers/run_32k", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32_768)
    parser.add_argument(
        "--target_tokens", type=int, default=None, help="Max tokens to train on (default: use all data)"
    )
    parser.add_argument("--val_docs", type=int, default=10_000, help="Max val docs to evaluate on")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=["google/gemma-3-1b-it", "openai/gpt-oss-20b"],
        help="HuggingFace model IDs for comparison",
    )
    parser.add_argument("--external_dir", default="data/tokenizers/external")
    parser.add_argument("--wandb_project", default="gemmeh-2024")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = HAS_WANDB and not args.no_wandb
    config = {
        "vocab_size": args.vocab_size,
        "target_tokens": args.target_tokens,
        "input": args.input,
        "model_type": "bpe",
        "byte_fallback": True,
        "split_digits": True,
        "normalization": "identity",
        "special_tokens": USER_DEFINED_SYMBOLS,
    }

    # Step 1: External tokenizers
    external_paths = {}
    if args.compare and HAS_TRANSFORMERS and HAS_HF_HUB:
        external_paths = ensure_external_tokenizers(args.compare, args.external_dir)

    # Step 2: Train
    tok_label = f"{args.target_tokens / 1e9:.0f}B" if args.target_tokens else "all"
    print(f"\n[1/3] Training SentencePiece BPE (vocab={args.vocab_size:,}, tokens={tok_label})...")
    train_sentencepiece(args.input, args.output_dir, args.vocab_size, args.target_tokens)

    # Step 3: Evaluate
    print("\n[2/3] Evaluating...")
    tokenizers = load_tokenizers(args.output_dir, external_paths)
    val_docs = load_val_docs(args.val, args.val_docs)

    all_metrics = {}
    for name, tok in tokenizers.items():
        all_metrics[name] = evaluate_tokenizer(tok, name, val_docs)
    rt = roundtrip_check(tokenizers["ours"], val_docs)

    # Step 4: Log
    print("\n[3/3] Logging...")
    if use_wandb:
        log_to_wandb(config, all_metrics, rt, tokenizers, val_docs, args.output_dir, args.wandb_project)
        print(f"  Logged to wandb project: {args.wandb_project}")
    else:
        for name, m in all_metrics.items():
            print(
                f"  {name}: {m['bytes_per_token']:.2f} bytes/tok, "
                f"{m['tokens_per_word']:.2f} tok/word, vocab_used={m['vocab_used_pct']:.1f}%"
            )
        print(f"  Roundtrip failures: {rt['failures']}/{len(val_docs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
