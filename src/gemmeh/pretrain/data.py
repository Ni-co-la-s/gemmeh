"""
Data loader for pre-tokenized binary files (uint16 memmap).

Reads fixed-length windows from a flat token array. Documents are separated
by eos tokens embedded in the stream — no special handling needed.
No seed is fixed in the dataloader, so that restarting training would not lead to the same data being seen.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


class MemmapDataset(Dataset):
    """
    Random-access dataset over a flat uint16 memmap file.

    Each sample is a contiguous window of (seq_len + 1) tokens:
    - input_ids = tokens[:-1]
    - targets   = tokens[1:]

    """

    def __init__(self, path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        # Number of valid starting positions for a window of seq_len+1
        self.num_samples = self.num_tokens - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def create_dataloader(
    path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    steps_per_epoch: int | None = None,
) -> DataLoader:
    """Create a DataLoader for training or validation.
    We use RandomSampler(replacement=True), which samples random offsets on the fly without materializing all indices.
    """
    dataset = MemmapDataset(path, seq_len)

    # Number of samples drawn per logical epoch. Defaults to one full pass.
    num_samples = (steps_per_epoch * batch_size) if steps_per_epoch is not None else len(dataset)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=num_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # avoid ragged batches
    )
