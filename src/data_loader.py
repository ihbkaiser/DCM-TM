"""
Data loader for preprocessed BoW corpora with timestamp information.

Loads sparse BoW matrices, vocabulary, and timestamps.
Provides per-timestamp iterators for streaming continual learning.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class TimestampData:
    """Data for a single timestamp."""
    timestamp: int
    bow: sp.csr_matrix          # (n_docs, vocab_size) sparse BoW
    label: str = ""             # e.g. "1987-1989"
    n_docs: int = 0
    vocab_size: int = 0

    def __post_init__(self):
        self.n_docs, self.vocab_size = self.bow.shape


@dataclass
class Corpus:
    """Full corpus with train/test splits and per-timestamp access."""
    vocab: list[str]
    vocab_size: int
    timestamps: list[int]              # sorted unique timestamps
    time_labels: dict[int, str]        # timestamp → year range string
    train_data: dict[int, TimestampData]   # timestamp → TimestampData
    test_data: Optional[dict[int, TimestampData]] = None


class BoWDataset(Dataset):
    """PyTorch dataset wrapping a sparse BoW matrix."""

    def __init__(self, bow: sp.csr_matrix):
        self.bow = bow
        self.n_docs, self.vocab_size = bow.shape

    def __len__(self):
        return self.n_docs

    def __getitem__(self, idx):
        # Convert sparse row to dense tensor
        row = self.bow[idx].toarray().flatten().astype(np.float32)
        return torch.from_numpy(row)


# ─── Loading Functions ────────────────────────────────────────────────────────

def load_corpus(data_dir: str) -> Corpus:
    """Load preprocessed corpus from a directory.

    Expected files:
        train_bow.npz, test_bow.npz   — sparse BoW matrices
        vocab.txt                       — one word per line
        train_times.txt, test_times.txt — one timestamp per line
        time2id.txt                     — "id\\tyear_range" per line
    """
    data_dir = Path(data_dir)

    # Vocabulary
    vocab = (data_dir / "vocab.txt").read_text().strip().split("\n")
    vocab_size = len(vocab)

    # Time labels
    time_labels = {}
    time2id_path = data_dir / "time2id.txt"
    if time2id_path.exists():
        for line in time2id_path.read_text().strip().split("\n"):
            parts = line.strip().split("\t")
            time_labels[int(parts[0])] = parts[1]

    # Train BoW + timestamps
    train_bow = sp.load_npz(str(data_dir / "train_bow.npz"))
    train_times = np.loadtxt(str(data_dir / "train_times.txt"), dtype=int)

    # Test BoW + timestamps
    test_bow_path = data_dir / "test_bow.npz"
    test_times_path = data_dir / "test_times.txt"
    has_test = test_bow_path.exists() and test_times_path.exists()
    if has_test:
        test_bow = sp.load_npz(str(test_bow_path))
        test_times = np.loadtxt(str(test_times_path), dtype=int)

    # Group by timestamp
    unique_ts = sorted(set(train_times.tolist()))

    train_data = {}
    for ts in unique_ts:
        mask = train_times == ts
        train_data[ts] = TimestampData(
            timestamp=ts,
            bow=train_bow[mask],
            label=time_labels.get(ts, str(ts)),
        )

    test_data = None
    if has_test:
        test_data = {}
        for ts in unique_ts:
            mask = test_times == ts
            if mask.sum() > 0:
                test_data[ts] = TimestampData(
                    timestamp=ts,
                    bow=test_bow[mask],
                    label=time_labels.get(ts, str(ts)),
                )

    print(f"Loaded corpus: {vocab_size} vocab, {len(unique_ts)} timestamps, "
          f"{train_bow.shape[0]} train docs" +
          (f", {test_bow.shape[0]} test docs" if has_test else ""))

    return Corpus(
        vocab=vocab,
        vocab_size=vocab_size,
        timestamps=unique_ts,
        time_labels=time_labels,
        train_data=train_data,
        test_data=test_data,
    )


def make_dataloader(
    ts_data: TimestampData,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for a single timestamp's data."""
    dataset = BoWDataset(ts_data.bow)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
