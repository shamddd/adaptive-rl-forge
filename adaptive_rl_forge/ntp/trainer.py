"""
Next-Token Prediction (NTP) training engine.
"""

import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple


def train_ntp_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Executes a single NTP pre-training step.
    Returns: (loss, elapsed_seconds, num_tokens_processed)
    """
    model.train()
    start = time.time()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()
    _, loss = model(input_ids=input_ids, labels=labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    elapsed = time.time() - start
    tokens_processed = input_ids.numel()
    return loss.item(), elapsed, tokens_processed
