"""
Standardized compute accounting & metrics tracking module.
"""

import time
import torch
from typing import Dict, Any


class ComputeTracker:
    """
    Tracks FLOPs, tokens, execution time, and memory usage across training steps.
    """

    def __init__(self, model_num_params: int):
        self.model_num_params = model_num_params
        self.start_time = time.time()

        self.ntp_tokens = 0
        self.sft_tokens = 0
        self.rl_generated_tokens = 0
        self.total_steps = 0
        self.step_history = []

    def record_step(
        self,
        step: int,
        objective_type: str,
        loss: float,
        tokens_processed: int,
        elapsed_seconds: float,
        extra_metrics: Dict[str, Any] = None,
    ):
        self.total_steps += 1
        if objective_type == "NTP":
            self.ntp_tokens += tokens_processed
        elif objective_type == "SFT":
            self.sft_tokens += tokens_processed
        elif objective_type == "RL":
            self.rl_generated_tokens += tokens_processed

        # Standard 6 * N * tokens approximation for training FLOPs
        flops = 6 * self.model_num_params * tokens_processed

        record = {
            "step": step,
            "objective_type": objective_type,
            "loss": loss,
            "tokens": tokens_processed,
            "flops": flops,
            "elapsed_seconds": elapsed_seconds,
        }
        if extra_metrics:
            record.update(extra_metrics)
        self.step_history.append(record)

    def summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        total_tokens = self.ntp_tokens + self.sft_tokens + self.rl_generated_tokens
        total_flops = 6 * self.model_num_params * total_tokens
        cpu_gpu_hours = total_time / 3600.0

        return {
            "total_steps": self.total_steps,
            "total_tokens": total_tokens,
            "ntp_tokens": self.ntp_tokens,
            "sft_tokens": self.sft_tokens,
            "rl_generated_tokens": self.rl_generated_tokens,
            "total_flops": total_flops,
            "wall_clock_seconds": total_time,
            "compute_hours": cpu_gpu_hours,
        }
