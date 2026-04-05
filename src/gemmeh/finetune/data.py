"""
Dataloader for OpenHermes instruction data with on-the-fly tokenization.
https://huggingface.co/datasets/teknium/openhermes

Input JSON format (list of objects):
    [
        {
            "instruction": "...",
            "input": "",          # optional extra context, mostly empty
            "output": "..."
        },
        ...
    ]

Chat template:
    <start_of_turn>user
    {instruction}[\\n{input}]
    <end_of_turn>
    <start_of_turn>model
    {output}<end_of_turn>

Only assistant tokens contribute to the loss to prevent the model
from wasting capacity learning to predict the user's question.

Sequences longer than max_seq_len are truncated from the right.
"""

import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


# Template builder


def build_prompt_and_response(instruction: str, input_: str, output: str) -> tuple[str, str]:
    """
    Return (prompt_str, response_str) separately so we can find the
    boundary between them after tokenization.

    prompt_str  = everything up to and including <start_of_turn>model\\n
    response_str = the assistant's reply + closing <end_of_turn>
    """
    user_content = instruction.strip()
    if input_ and input_.strip():
        user_content = user_content + "\n" + input_.strip()

    prompt = f"<start_of_turn>user\n{user_content}\n<end_of_turn>\n<start_of_turn>model\n"
    response = f"{output.strip()}<end_of_turn>"
    return prompt, response


# Dataset


class ChatDataset(Dataset):
    """
    Tokenizes OpenHermes examples on the fly.

    Each item returns:
        input_ids : [seq_len]
        targets   : [seq_len]  — same as input_ids shifted by 1,
                                    with prompt positions set to -1
        loss_mask : [seq_len]  — True where loss should be computed
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int,
    ):
        self.examples = examples
        self.sp = tokenizer
        self.max_seq_len = max_seq_len

        # Resolve special token IDs once
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        prompt, response = build_prompt_and_response(
            ex["instruction"],
            ex.get("input", ""),
            ex["output"],
        )

        # Tokenize prompt and response separately to find boundary
        prompt_ids = self.sp.encode(prompt, out_type=int)
        response_ids = self.sp.encode(response, out_type=int)
        response_ids.append(self.eos_id)  # explicit EOS after <end_of_turn>

        full_ids = prompt_ids + response_ids

        # Truncate to max_seq_len + 1 (we need n+1 tokens to produce n pairs)
        max_total = self.max_seq_len + 1
        if len(full_ids) > max_total:
            full_ids = full_ids[:max_total]

        # Build input / target pairs
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        targets = torch.tensor(full_ids[1:], dtype=torch.long)

        # Loss mask: 1 only for response tokens in the target positions.
        prompt_len = min(len(prompt_ids), len(input_ids))
        loss_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        loss_mask[prompt_len:] = True

        # Uf no response tokens made it pass truncation
        if not loss_mask.any():
            # Get the next example small enough
            return self.__getitem__((idx + 1) % len(self.examples))

        # Apply mask to targets: set prompt target positions to -1
        targets = targets.clone()
        targets[~loss_mask] = -1

        return input_ids, targets, loss_mask


# Collator pads variable-length sequences to batch max


class ChatCollator:
    """
    Pads sequences in a batch to the length of the longest example.

    Padding is applied on the right. Padded target positions are set to -1
    so they are excluded from the loss. loss_mask is padded with False.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_id = pad_token_id

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids_list, targets_list, masks_list = zip(*batch)
        max_len = max(x.shape[0] for x in input_ids_list)

        padded_ids = []
        padded_tgt = []
        padded_msk = []

        for ids, tgt, msk in zip(input_ids_list, targets_list, masks_list):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_id, dtype=torch.long)]))
            padded_tgt.append(torch.cat([tgt, torch.full((pad_len,), -1, dtype=torch.long)]))
            padded_msk.append(torch.cat([msk, torch.zeros(pad_len, dtype=torch.bool)]))

        return (
            torch.stack(padded_ids),
            torch.stack(padded_tgt),
            torch.stack(padded_msk),
        )


def load_openhermes(path: str) -> list[dict]:
    """Load OpenHermes JSON"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filter out any malformed examples
    data = [ex for ex in data if ex.get("instruction", "").strip() and ex.get("output", "").strip()]
    print(f"Loaded {len(data):,} examples from {path}")
    return data


def make_splits(
    data: list[dict],
    val_fraction: float = 0.02,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train / val."""
    if shuffle:
        rng = random.Random(seed)
        data = data[:]
        rng.shuffle(data)
    n_val = max(1, int(len(data) * val_fraction))
    return data[n_val:], data[:n_val]


def create_chat_dataloaders(
    data_path: str,
    tokenizer: spm.SentencePieceProcessor,
    max_seq_len: int,
    micro_batch_size: int,
    val_fraction: float = 0.02,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build train and val DataLoaders from an OpenHermes JSON file.
    """
    data = load_openhermes(data_path)
    train_data, val_data = make_splits(data, val_fraction, shuffle, seed)
    print(f"  Train: {len(train_data):,} examples | Val: {len(val_data):,} examples")

    collator = ChatCollator(pad_token_id=0)

    train_ds = ChatDataset(train_data, tokenizer, max_seq_len)
    val_ds = ChatDataset(val_data, tokenizer, max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, len(train_data)
