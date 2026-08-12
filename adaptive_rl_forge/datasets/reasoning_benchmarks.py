"""
Reasoning benchmarks dataset module for SFT and RL verification tasks.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple


class ArithmeticReasoningDataset(Dataset):
    """
    Verifiable arithmetic reasoning dataset (e.g. 'A + B = C').
    Provides prompt tokens, target tokens, and exact answer verification keys.
    """

    def __init__(
        self,
        num_samples: int = 500,
        vocab_size: int = 1000,
        max_num: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        rng = random.Random(seed)

        self.samples = []
        for _ in range(num_samples):
            a = rng.randint(1, max_num)
            b = rng.randint(1, max_num)
            op = rng.choice(["+", "-"])
            ans = a + b if op == "+" else a - b

            # Token encoding scheme:
            # 1: PAD, 2: BOS, 3: EOS, 4: '+', 5: '-', 6: '='
            # Numbers mapped to offset 10: 10 + n
            op_tok = 4 if op == "+" else 5
            prompt_tokens = [2, 10 + a, op_tok, 10 + b, 6]
            ans_token = 10 + (ans % 900)  # mapped to valid vocab range
            full_sequence = prompt_tokens + [ans_token, 3]

            self.samples.append({
                "prompt_ids": torch.tensor(prompt_tokens, dtype=torch.long),
                "full_ids": torch.tensor(full_sequence, dtype=torch.long),
                "answer_token": ans_token,
                "a": a,
                "b": b,
                "op": op,
                "ans": ans,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(b["full_ids"]) for b in batch)
    padded_full = []
    padded_prompt = []
    max_prompt_len = max(len(b["prompt_ids"]) for b in batch)

    answers = []
    for b in batch:
        f = b["full_ids"]
        p = b["prompt_ids"]
        f_pad = torch.cat([f, torch.ones(max_len - len(f), dtype=torch.long)])
        p_pad = torch.cat([p, torch.ones(max_prompt_len - len(p), dtype=torch.long)])
        padded_full.append(f_pad)
        padded_prompt.append(p_pad)
        answers.append(b["answer_token"])

    return {
        "full_ids": torch.stack(padded_full),
        "prompt_ids": torch.stack(padded_prompt),
        "answer_tokens": torch.tensor(answers, dtype=torch.long),
    }


def get_reasoning_dataloader(
    num_samples: int = 500,
    batch_size: int = 16,
    seed: int = 42,
) -> DataLoader:
    ds = ArithmeticReasoningDataset(num_samples=num_samples, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
