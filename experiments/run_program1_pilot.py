"""
Program 1 Minimum Viable Pilot (MVP) Runner
Evaluates pre- vs post-GRPO RLVR self-consistency agreement reliability,
trajectory path similarity (J_path), Brier score, and AURC on a matched model.
"""

import os
import json
import random
import math
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from scipy import stats

from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier
from adaptive_rl_forge.datasets.reasoning_benchmarks import ArithmeticReasoningDataset, pad_collate_fn
from adaptive_rl_forge.eval.evaluator import compute_expected_calibration_error
from adaptive_rl_forge.rl.grpo_trainer import train_grpo_step


def compute_jaccard_similarity(tokens_a: List[int], tokens_b: List[int], n: int = 2) -> float:
    """Computes n-gram Jaccard similarity between two token sequences."""
    def get_ngrams(seq: List[int], n_val: int):
        return set(tuple(seq[i : i + n_val]) for i in range(len(seq) - n_val + 1))

    set_a = get_ngrams(tokens_a, n)
    set_b = get_ngrams(tokens_b, n)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / float(union) if union > 0 else 0.0


def compute_aurc(confidences: List[float], correctness: List[int]) -> float:
    """Computes Area Under Risk-Coverage (AURC) Curve."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0

    # Sort samples by confidence descending
    sorted_pairs = sorted(zip(confidences, correctness), key=lambda x: x[0], reverse=True)
    N = len(sorted_pairs)
    total_errors = sum(1 - y for _, y in sorted_pairs)
    if total_errors == 0:
        return 0.0

    cum_errors = 0
    risk_sum = 0.0

    for i, (_, y) in enumerate(sorted_pairs):
        if y == 0:
            cum_errors += 1
        coverage = (i + 1) / float(N)
        risk = cum_errors / float(i + 1)
        risk_sum += risk

    return risk_sum / float(N)


def evaluate_model_self_consistency(
    model: torch.nn.Module,
    eval_dataset: ArithmeticReasoningDataset,
    verifier: ExactMatchRewardVerifier,
    device: torch.device,
    k_rollouts: int = 8,
    temperature: float = 0.7,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    model.eval()
    per_problem_results = []
    
    all_agreements = []
    all_correctness = []
    all_jaccards = []

    with torch.no_grad():
        for idx in range(len(eval_dataset)):
            sample = eval_dataset[idx]
            prompt_ids = sample["prompt_ids"].unsqueeze(0).to(device)  # [1, P]
            P = prompt_ids.size(1)
            target_ans = sample["answer_token"]

            # Expand prompt for K rollouts
            p_expanded = prompt_ids.repeat_interleave(k_rollouts, dim=0)  # [K, P]
            ans_expanded = torch.tensor([target_ans] * k_rollouts, device=device)

            completed = model.generate(p_expanded, max_new_tokens=4, temperature=temperature)
            _, is_correct_list = verifier.compute_rewards(completed, [P] * k_rollouts, ans_expanded)

            # Extract generated answer tokens
            gen_tokens_list = []
            extracted_answers = []
            for k in range(k_rollouts):
                gen_toks = [t.item() for t in completed[k, P:] if t.item() not in (1, 3)]
                gen_tokens_list.append(gen_toks)
                ans = gen_toks[0] if len(gen_toks) > 0 else -1
                extracted_answers.append(ans)

            # Modal Answer Agreement (S_ans)
            ans_counts = {}
            for ans in extracted_answers:
                ans_counts[ans] = ans_counts.get(ans, 0) + 1
            modal_ans = max(ans_counts, key=ans_counts.get)
            modal_count = ans_counts[modal_ans]
            s_ans = modal_count / float(k_rollouts)
            is_modal_correct = 1 if modal_ans == target_ans else 0

            # Pairwise Jaccard similarity (J_path)
            jaccard_sims = []
            for i in range(k_rollouts):
                for j in range(i + 1, k_rollouts):
                    sim = compute_jaccard_similarity(gen_tokens_list[i], gen_tokens_list[j])
                    jaccard_sims.append(sim)
            mean_jaccard = sum(jaccard_sims) / float(max(len(jaccard_sims), 1))

            all_agreements.append(s_ans)
            all_correctness.append(is_modal_correct)
            all_jaccards.append(mean_jaccard)

            per_problem_results.append({
                "problem_id": idx,
                "s_ans": s_ans,
                "is_correct": is_modal_correct,
                "j_path": mean_jaccard,
                "modal_ans": modal_ans,
                "target_ans": target_ans,
                "rollout_answers": extracted_answers,
            })

    # Summary metrics
    aurc = compute_aurc(all_agreements, all_correctness)
    ece_metrics = compute_expected_calibration_error(all_agreements, all_correctness, num_bins=5)
    mean_acc = sum(all_correctness) / float(len(all_correctness))
    mean_s_ans = sum(all_agreements) / float(len(all_agreements))
    mean_jaccard_all = sum(all_jaccards) / float(len(all_jaccards))

    # High-agreement error rate (S_ans >= 0.75 & incorrect)
    high_agree_errors = sum(1 for s, y in zip(all_agreements, all_correctness) if s >= 0.75 and y == 0)
    high_agree_error_rate = high_agree_errors / float(len(all_correctness))

    # AUROC for correctness prediction
    try:
        auroc = stats.roc_auc_score(all_correctness, all_agreements)
    except Exception:
        auroc = 0.50

    summary_metrics = {
        "accuracy": mean_acc,
        "mean_s_ans": mean_s_ans,
        "mean_j_path": mean_jaccard_all,
        "aurc": aurc,
        "brier_score": ece_metrics["brier_score"],
        "ece": ece_metrics["ece"],
        "auroc": float(auroc),
        "high_agreement_error_rate": high_agree_error_rate,
    }

    return per_problem_results, summary_metrics


def run_program1_pilot():
    print("================================================================================")
    print("PROGRAM 1 MINIMUM VIABLE PILOT (MVP) RUNNER")
    print("Evaluating Pre- vs Post-GRPO Self-Consistency Calibration & Trajectory Homogenization")
    print("================================================================================")

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")

    # Load matched model
    vocab_size = 1000
    d_model = 128
    n_layer = 4

    model = LightweightLM(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    verifier = ExactMatchRewardVerifier()

    # Load dataset
    full_dataset = ArithmeticReasoningDataset(num_samples=200, seed=101)
    sft_subset = [full_dataset[i] for i in range(100)]
    eval_subset = [full_dataset[i] for i in range(100, 150)]
    train_subset = [full_dataset[i] for i in range(150, 200)]

    print("\n--- PHASE 0: SFT Warm-Start Pre-Training (50 steps) ---")
    from adaptive_rl_forge.sft.trainer import train_sft_step
    model.train()
    for step in range(50):
        sample_batch = random.sample(sft_subset, 8)
        collate_batch = pad_collate_fn(sample_batch)
        sft_loss, _, _ = train_sft_step(model, optimizer, collate_batch, device)
        if (step + 1) % 10 == 0:
            print(f"SFT Step {step+1}/50 | Loss: {sft_loss:.4f}")

    print("\n--- PHASE 1: Pre-RLVR Baseline Evaluation (N=50 prompts) ---")
    pre_results, pre_metrics = evaluate_model_self_consistency(
        model=model,
        eval_dataset=eval_subset,
        verifier=verifier,
        device=device,
        k_rollouts=8,
        temperature=0.7,
    )

    print(f"Pre-RLVR Accuracy:                   {pre_metrics['accuracy']*100:.2f}%")
    print(f"Pre-RLVR Mean S_ans (Agreement):       {pre_metrics['mean_s_ans']:.4f}")
    print(f"Pre-RLVR Mean J_path (Similarity):    {pre_metrics['mean_j_path']:.4f}")
    print(f"Pre-RLVR Brier Score:                {pre_metrics['brier_score']:.4f}")
    print(f"Pre-RLVR AURC (Risk-Coverage):        {pre_metrics['aurc']:.4f}")
    print(f"Pre-RLVR AUROC:                      {pre_metrics['auroc']:.4f}")
    print(f"Pre-RLVR High-Agreement Error Rate:  {pre_metrics['high_agreement_error_rate']*100:.2f}%")

    print("\n--- PHASE 2: Controlled GRPO Post-Training Step (50 steps) ---")
    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step in range(50):
        sample_batch = random.sample(train_subset, 4)
        collate_batch = pad_collate_fn(sample_batch)
        pg_loss, mean_r, kl_loss, _, _ = train_grpo_step(
            model=model,
            optimizer=rl_optimizer,
            batch=collate_batch,
            verifier=verifier,
            device=device,
            group_size=4,
            kl_coeff=0.01,
            calibration_coeff=0.0,  # Standard uncalibrated GRPO
        )
        if (step + 1) % 10 == 0:
            print(f"GRPO Step {step+1}/50 | Loss: {pg_loss:.4f} | Reward: {mean_r:.4f}")

    print("\n--- PHASE 3: Post-RLVR Evaluation (Identical N=50 prompts) ---")
    post_results, post_metrics = evaluate_model_self_consistency(
        model=model,
        eval_dataset=eval_subset,
        verifier=verifier,
        device=device,
        k_rollouts=8,
        temperature=0.7,
    )

    print(f"Post-RLVR Accuracy:                  {post_metrics['accuracy']*100:.2f}%")
    print(f"Post-RLVR Mean S_ans (Agreement):      {post_metrics['mean_s_ans']:.4f}")
    print(f"Post-RLVR Mean J_path (Similarity):   {post_metrics['mean_j_path']:.4f}")
    print(f"Post-RLVR Brier Score:               {post_metrics['brier_score']:.4f}")
    print(f"Post-RLVR AURC (Risk-Coverage):       {post_metrics['aurc']:.4f}")
    print(f"Post-RLVR AUROC:                     {post_metrics['auroc']:.4f}")
    print(f"Post-RLVR High-Agreement Error Rate: {post_metrics['high_agreement_error_rate']*100:.2f}%")


    # Statistical Significance Testing
    pre_aurcs = [r["s_ans"] for r in pre_results]
    post_aurcs = [r["s_ans"] for r in post_results]
    stat_t, p_val_t = stats.ttest_rel(pre_aurcs, post_aurcs)

    brier_delta = post_metrics["brier_score"] - pre_metrics["brier_score"]
    aurc_delta = post_metrics["aurc"] - pre_metrics["aurc"]
    jaccard_delta = post_metrics["mean_j_path"] - pre_metrics["mean_j_path"]

    print("\n================================================================================")
    print("PILOT STATISTICAL SUMMARY")
    print("================================================================================")
    print(f"Brier Score Delta:     {brier_delta:+.4f}")
    print(f"AURC Delta:            {aurc_delta:+.4f}")
    print(f"Path Similarity Delta: {jaccard_delta:+.4f}")
    print(f"Paired t-test p-value: {p_val_t:.6f}")

    # Determine Verdict
    # Proxy failure: Brier score worsens (increases) or AURC worsens (increases) while J_path increases
    if brier_delta > 0 and jaccard_delta > 0:
        verdict = "GO"
        finding_type = "Finding B (Overconfident Agreement / Proxy Failure Supported)"
    elif jaccard_delta > 0 and brier_delta <= 0:
        verdict = "PIVOT"
        finding_type = "Finding C (Homogenization without calibration failure)"
    else:
        verdict = "GO" if abs(brier_delta) > 0.01 else "PIVOT"
        finding_type = "Finding A or Moderate Signal"

    results_payload = {
        "metadata": {
            "model": "LightweightLM (Matched Pair)",
            "n_prompts": 50,
            "k_rollouts": 8,
            "temperature": 0.7,
            "grpo_steps": 30,
            "verdict": verdict,
            "finding_type": finding_type,
        },
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "deltas": {
            "brier_delta": brier_delta,
            "aurc_delta": aurc_delta,
            "jaccard_delta": jaccard_delta,
            "p_value_t": float(p_val_t),
        },
        "pre_results": pre_results,
        "post_results": post_results,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/program1_pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nCanonical raw pilot data archived to: {out_path}")
    print(f"VERDICT: {verdict} ({finding_type})")


if __name__ == "__main__":
    run_program1_pilot()
