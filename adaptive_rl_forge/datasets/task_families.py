"""
Multi-Task Family Benchmarks Module.
Provides 3 verifiable task families:
1. Arithmetic Reasoning
2. Symbolic Logic
3. Code Execution Syntax & Output Verification
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple


class TaskFamilyDataset(Dataset):
    """
    Dataset generator supporting 3 verifiable task categories.
    """

    def __init__(
        self,
        task_family: str = "arithmetic",
        num_samples: int = 200,
        tokenizer: Any = None,
        seed: int = 42,
    ):
        super().__init__()
        self.task_family = task_family
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        rng = random.Random(seed)

        self.samples = []

        for _ in range(num_samples):
            if task_family == "arithmetic":
                a = rng.randint(10, 99)
                b = rng.randint(10, 99)
                op = rng.choice(["+", "*"])
                ans = a + b if op == "+" else a * b
                prompt = f"Q: What is {a} {op} {b}? A:"
                target = f" {ans}"

            elif task_family == "logic":
                start = rng.randint(1, 20)
                step = rng.randint(2, 5)
                seq = [start + i * step for i in range(4)]
                ans = start + 4 * step
                seq_str = ", ".join(map(str, seq))
                prompt = f"Q: Continue sequence {seq_str}, ? A:"
                target = f" {ans}"

            else:  # code execution
                val = rng.randint(1, 30)
                ans = val * 2
                prompt = f"def double(x): return x * 2\n# Output for double({val})\nResult = "
                target = f"{ans}"

            if tokenizer is not None:
                p_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
                full_ids = tokenizer.encode(prompt + target, return_tensors="pt")[0]
            else:
                p_ids = torch.tensor([1, 2, 3])
                full_ids = torch.tensor([1, 2, 3, 4])

            self.samples.append({
                "prompt": prompt,
                "target": target,
                "prompt_ids": p_ids,
                "full_ids": full_ids,
                "ground_truth": target.strip(),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def task_family_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_p = max(len(b["prompt_ids"]) for b in batch)
    max_f = max(len(b["full_ids"]) for b in batch)

    p_padded = []
    f_padded = []
    truths = []

    for b in batch:
        p = b["prompt_ids"]
        f = b["full_ids"]
        p_pad = torch.cat([p, torch.zeros(max_p - len(p), dtype=torch.long)])
        f_pad = torch.cat([f, torch.zeros(max_f - len(f), dtype=torch.long)])
        p_padded.append(p_pad)
        f_padded.append(f_pad)
        truths.append(b["ground_truth"])

    return {
        "prompt_ids": torch.stack(p_padded),
        "full_ids": torch.stack(f_padded),
        "ground_truths": truths,
    }


def get_task_family_dataloader(
    task_family: str = "arithmetic",
    num_samples: int = 100,
    batch_size: int = 8,
    tokenizer: Any = None,
    seed: int = 42,
) -> DataLoader:
    ds = TaskFamilyDataset(task_family, num_samples, tokenizer, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=task_family_collate)
