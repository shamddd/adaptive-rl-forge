"""
Checkpoint tracker for saving, loading, and snapshotting model training states.
"""

import os
import torch
from typing import Dict, Any, Optional


class CheckpointTracker:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        if filename is None:
            filename = f"checkpoint_step_{step}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        return path

    def load_checkpoint(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt
