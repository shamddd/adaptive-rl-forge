"""
Evaluation module for model evaluation across tasks, solution diversity, and retention.
"""

import math
import torch
from typing import Dict, Any, List
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier


def evaluate_model_capabilities(
    model: torch.nn.Module,
    eval_dataloader: Any,
    verifier: ExactMatchRewardVerifier,
    device: torch.device,
    k_list: List[int] = [1, 4],
) -> Dict[str, float]:
    """
    Evaluates pass@1, pass@4, accuracy, and solution diversity on evaluation dataloader.
    """
    model.eval()
    total_eval_samples = 0
    correct_pass1 = 0
    correct_pass4 = 0

    unique_solution_hashes = set()

    with torch.no_grad():
        for batch in eval_dataloader:
            prompt_ids = batch["prompt_ids"].to(device)
            answer_tokens = batch["answer_tokens"].to(device)
            B, P = prompt_ids.size()
            total_eval_samples += B

            # Pass@1
            completed_1 = model.generate(prompt_ids, max_new_tokens=4, temperature=0.1)
            _, is_correct_1 = verifier.compute_rewards(completed_1, [P] * B, answer_tokens)
            correct_pass1 += sum(is_correct_1)

            # Record generated output tokens for solution diversity
            for i in range(B):
                gen_seq = tuple(completed_1[i, P:].tolist())
                unique_solution_hashes.add(gen_seq)

            # Pass@4
            p_expanded = prompt_ids.repeat_interleave(4, dim=0)
            ans_expanded = answer_tokens.repeat_interleave(4, dim=0)
            completed_4 = model.generate(p_expanded, max_new_tokens=4, temperature=0.8)
            _, is_correct_4 = verifier.compute_rewards(completed_4, [P] * 4 * B, ans_expanded)

            for i in range(B):
                sample_corrects = is_correct_4[i * 4 : (i + 1) * 4]
                if any(sample_corrects):
                    correct_pass4 += 1

    pass1_acc = correct_pass1 / float(max(total_eval_samples, 1))
    pass4_acc = correct_pass4 / float(max(total_eval_samples, 1))
    diversity_ratio = len(unique_solution_hashes) / float(max(total_eval_samples, 1))

    return {
        "pass_at_1": pass1_acc,
        "pass_at_4": pass4_acc,
        "solution_diversity_ratio": diversity_ratio,
    }
