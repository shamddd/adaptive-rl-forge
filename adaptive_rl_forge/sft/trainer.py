"""
Supervised Fine-Tuning (SFT) training engine.
"""

import time
import torch
from typing import Dict, Any, Tuple


def train_sft_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Executes a single SFT step on prompt-answer pairs.
    Returns: (loss, elapsed_seconds, num_tokens_processed)
    """
    model.train()
    start = time.time()

    full_ids = batch["full_ids"].to(device)

    optimizer.zero_grad()
    _, loss = model(input_ids=full_ids, labels=full_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    elapsed = time.time() - start
    tokens_processed = full_ids.numel()
    return loss.item(), elapsed, tokens_processed
