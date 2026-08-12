"""
Group Relative Policy Optimization (GRPO) & REINFORCE Policy Gradient engine for RL excursions.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier


def train_grpo_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    verifier: ExactMatchRewardVerifier,
    device: torch.device,
    group_size: int = 4,
    kl_coeff: float = 0.05,
    ref_model: Optional[torch.nn.Module] = None,
) -> Tuple[float, float, float, int, float]:
    """
    Executes a GRPO step:
    For each prompt in batch, generates G completions, computes group-normalized rewards & KL divergence.
    Returns: (pg_loss, mean_reward, kl_div, num_generated_tokens, elapsed_seconds)
    """
    model.train()
    start = time.time()

    prompt_ids = batch["prompt_ids"].to(device)  # [B, P]
    answer_tokens = batch["answer_tokens"].to(device)  # [B]
    B, P = prompt_ids.size()

    # Expand prompt batch for group sampling
    prompt_expanded = prompt_ids.repeat_interleave(group_size, dim=0)  # [B*G, P]
    answer_expanded = answer_tokens.repeat_interleave(group_size, dim=0)

    # Rollout generation
    with torch.no_grad():
        completed = model.generate(prompt_expanded, max_new_tokens=4, temperature=0.8)

    prompt_lens = [P] * (B * group_size)
    rewards, _ = verifier.compute_rewards(completed, prompt_lens, answer_expanded)
    rewards = rewards.view(B, group_size)

    # Group Relative Normalization
    r_mean = rewards.mean(dim=1, keepdim=True)
    r_std = rewards.std(dim=1, keepdim=True) + 1e-6
    advantages = ((rewards - r_mean) / r_std).view(-1)  # [B*G]

    # Evaluate Log Probabilities of completed tokens
    logits, _ = model(completed)
    gen_logits = logits[:, P - 1 : -1, :]  # Logits predicting response tokens
    gen_tokens = completed[:, P:]  # Generated response tokens

    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = token_log_probs.sum(dim=1)

    # Policy Gradient Loss
    pg_loss = -(advantages * seq_log_probs).mean()

    # KL Penalty relative to reference policy if provided
    kl_loss = torch.tensor(0.0, device=device)
    if ref_model is not None:
        with torch.no_grad():
            ref_logits, _ = ref_model(completed)
            ref_gen_logits = ref_logits[:, P - 1 : -1, :]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
        kl_loss = (token_log_probs - ref_token_log_probs).sum(dim=1).mean()
        pg_loss = pg_loss + kl_coeff * kl_loss

    optimizer.zero_grad()
    pg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    elapsed = time.time() - start
    gen_token_count = gen_tokens.numel()
    return pg_loss.item(), rewards.mean().item(), kl_loss.item(), gen_token_count, elapsed
