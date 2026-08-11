"""
Data loader for pre-tokenized binary files (uint16 memmap).

Reads fixed-length windows from a flat token array. Documents are separated
by eos tokens embedded in the stream — no special handling needed.
No seed is fixed in the training dataloader, so that restarting training would not lead to the same data being seen.
The validation dataloader retrieves non overlapping windows in sequential order so that it is consistent.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


class MemmapDataset(Dataset):
    """
    Access dataset over a flat uint16 memmap file.

    Each sample is a contiguous window of (seq_len + 1) tokens:
    - input_ids = tokens[:-1]
    - targets   = tokens[1:]

    """

    def __init__(self, path: str, seq_len: int, overlapping: bool = True):
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        self.overlapping = overlapping
        if overlapping:
            # Number of valid starting positions for a window of seq_len+1
            self.num_samples = self.num_tokens - seq_len
        else:
            # Number of valid starting positions without overlap
            self.num_samples = (self.num_tokens - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.overlapping:
            start = idx
        else:
            start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def create_train_dataloader(
    path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for training.
    We use RandomSampler(replacement=True), which samples random offsets on the fly without materializing all indices.
    """
    dataset = MemmapDataset(path, seq_len, True)

    # TODO: Use a fixed seed for reproducibility (would have to save generator state in checkpoint to not restart from beginning when resuming from checkpoint)
    sampler = RandomSampler(
        dataset,
        replacement=True,
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


def create_val_dataloader(
    path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for validation.
    We sample iteratively from the validation binary file, with no overlap.
    """
    dataset = MemmapDataset(path, seq_len, False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # avoid ragged batches
    )
