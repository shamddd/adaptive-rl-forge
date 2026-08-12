"""
Reward verifier functions for RL and pass@k evaluation.
"""

import torch
from typing import List, Dict, Any, Tuple


class ExactMatchRewardVerifier:
    """
    Verifies generated token responses against ground-truth answer tokens.
    Returns binary reward +1.0 for exact answer match, -0.1 for incorrect, and -0.5 for format invalidity.
    """

    def __init__(self, format_penalty: float = -0.5, incorrect_penalty: float = -0.1):
        self.format_penalty = format_penalty
        self.incorrect_penalty = incorrect_penalty

    def compute_rewards(
        self,
        generated_sequences: torch.Tensor,
        prompt_lengths: List[int],
        target_answer_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[bool]]:
        """
        Compute rewards for a batch of generated token sequences.
        """
        batch_size = generated_sequences.size(0)
        rewards = torch.zeros(batch_size, device=generated_sequences.device)
        is_correct_list = []

        for i in range(batch_size):
            p_len = prompt_lengths[i]
            gen_tokens = generated_sequences[i, p_len:]
            target = target_answer_tokens[i].item()

            # Find generated token before EOS (3) or PAD (1)
            valid_toks = [t.item() for t in gen_tokens if t.item() not in (1, 3)]
            if len(valid_toks) == 0:
                rewards[i] = self.format_penalty
                is_correct_list.append(False)
            elif valid_toks[0] == target:
                rewards[i] = 1.0
                is_correct_list.append(True)
            else:
                rewards[i] = self.incorrect_penalty
                is_correct_list.append(False)

        return rewards, is_correct_list
