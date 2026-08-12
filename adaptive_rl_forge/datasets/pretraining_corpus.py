"""
Pretraining corpus dataset generator for Next-Token Prediction (NTP).
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class SyntheticPretrainingDataset(Dataset):
    """
    Synthetic text token dataset simulating pre-training language distribution.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 64,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        g = torch.Generator().manual_seed(seed)
        # Create structured synthetic sequence data with repeated pattern n-grams
        raw_tokens = torch.randint(10, vocab_size, (num_samples, seq_len), generator=g)
        # Induce syntactic correlations (probabilistic next-token dependence)
        for i in range(1, seq_len):
            mask = (raw_tokens[:, i] % 3 == 0)
            raw_tokens[mask, i] = (raw_tokens[mask, i - 1] + 1) % vocab_size

        self.data = raw_tokens

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


def get_ntp_dataloader(
    num_samples: int = 1000,
    seq_len: int = 64,
    vocab_size: int = 1000,
    batch_size: int = 16,
    seed: int = 42,
) -> DataLoader:
    ds = SyntheticPretrainingDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
