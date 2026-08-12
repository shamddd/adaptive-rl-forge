"""
Diagnostic readiness signals module for checkpoint plasticity analysis.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


def compute_gradient_alignment(
    model: torch.nn.Module,
    ntp_batch: Dict[str, torch.Tensor],
    rl_batch: Dict[str, torch.Tensor],
    verifier: Any,
    device: torch.device,
) -> float:
    """
    Computes cosine similarity between NTP loss gradients and RL policy gradients:
    cos_sim = <g_NTP, g_RL> / (||g_NTP|| * ||g_RL||)
    """
    model.eval()

    # 1. Compute NTP gradients
    model.zero_grad()
    input_ids = ntp_batch["input_ids"].to(device)
    labels = ntp_batch["labels"].to(device)
    _, loss_ntp = model(input_ids=input_ids, labels=labels)
    loss_ntp.backward()

    g_ntp = []
    for p in model.parameters():
        if p.grad is not None:
            g_ntp.append(p.grad.detach().flatten())
    if not g_ntp:
        return 0.0
    v_ntp = torch.cat(g_ntp)

    # 2. Compute RL gradients
    model.zero_grad()
    prompt_ids = rl_batch["prompt_ids"].to(device)
    answer_tokens = rl_batch["answer_tokens"].to(device)
    B, P = prompt_ids.size()

    with torch.no_grad():
        completed = model.generate(prompt_ids, max_new_tokens=4, temperature=0.8)

    prompt_lens = [P] * B
    rewards, _ = verifier.compute_rewards(completed, prompt_lens, answer_tokens)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    logits, _ = model(completed)
    gen_logits = logits[:, P - 1 : -1, :]
    gen_tokens = completed[:, P:]

    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = token_log_probs.sum(dim=1)

    loss_rl = -(adv * seq_log_probs).mean()
    loss_rl.backward()

    g_rl = []
    for p in model.parameters():
        if p.grad is not None:
            g_rl.append(p.grad.detach().flatten())
    if not g_rl:
        return 0.0
    v_rl = torch.cat(g_rl)

    # 3. Compute Cosine Similarity
    norm_ntp = torch.norm(v_ntp)
    norm_rl = torch.norm(v_rl)

    if norm_ntp == 0 or norm_rl == 0:
        return 0.0

    cos_sim = torch.dot(v_ntp, v_rl) / (norm_ntp * norm_rl)
    model.zero_grad()
    return float(cos_sim.item())


def compute_policy_entropy(
    model: torch.nn.Module,
    eval_batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """
    Computes average per-token policy entropy H(pi).
    """
    model.eval()
    with torch.no_grad():
        input_ids = eval_batch["prompt_ids"].to(device)
        logits, _ = model(input_ids)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
    return float(entropy.item())


def compute_pass_at_k(
    model: torch.nn.Module,
    eval_batch: Dict[str, torch.Tensor],
    verifier: Any,
    device: torch.device,
    k: int = 4,
    temperature: float = 0.8,
) -> float:
    """
    Computes pass@k accuracy over evaluation samples.
    """
    model.eval()
    prompt_ids = eval_batch["prompt_ids"].to(device)
    answer_tokens = eval_batch["answer_tokens"].to(device)
    B, P = prompt_ids.size()

    passed = 0
    with torch.no_grad():
        for i in range(B):
            p_single = prompt_ids[i : i + 1].repeat(k, 1)
            ans_single = answer_tokens[i : i + 1].repeat(k)
            completed = model.generate(p_single, max_new_tokens=4, temperature=temperature)
            _, is_correct = verifier.compute_rewards(completed, [P] * k, ans_single)
            if any(is_correct):
                passed += 1

    return passed / float(B)


def extract_checkpoint_signals(
    model: torch.nn.Module,
    ntp_batch: Dict[str, torch.Tensor],
    rl_batch: Dict[str, torch.Tensor],
    verifier: Any,
    device: torch.device,
    step: int,
) -> Dict[str, float]:
    """
    Extracts complete dictionary of diagnostic pre-RL signals for a checkpoint.
    """
    grad_align = compute_gradient_alignment(model, ntp_batch, rl_batch, verifier, device)
    entropy = compute_policy_entropy(model, rl_batch, device)
    pass_k = compute_pass_at_k(model, rl_batch, verifier, device, k=4)

    return {
        "step": float(step),
        "gradient_alignment": grad_align,
        "policy_entropy": entropy,
        "pass_at_k": pass_k,
    }
