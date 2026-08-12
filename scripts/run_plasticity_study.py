"""
Script to execute the Checkpoint Plasticity Study (Phase 4 & Phase 5).
Collects pre-RL readiness signals across intermediate checkpoints and evaluates subsequent RL gain.
Saves outcome dataset to artifacts/plasticity/checkpoint_rl_outcomes.parquet.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np

from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.datasets.pretraining_corpus import get_ntp_dataloader
from adaptive_rl_forge.datasets.reasoning_benchmarks import get_reasoning_dataloader
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier
from adaptive_rl_forge.ntp.trainer import train_ntp_step
from adaptive_rl_forge.rl.grpo_trainer import train_grpo_step
from adaptive_rl_forge.carls.signals import extract_checkpoint_signals
from adaptive_rl_forge.eval.evaluator import evaluate_model_capabilities


def run_plasticity_study(output_path: str, seed: int = 42):
    print(f"=== Running Checkpoint Plasticity Study (Seed {seed}) ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    model = LightweightLM(vocab_size=1000, d_model=64, n_layer=4, n_head=4, max_seq_len=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    verifier = ExactMatchRewardVerifier()

    ntp_dl = get_ntp_dataloader(num_samples=500, seq_len=32, vocab_size=1000, batch_size=16, seed=seed)
    reason_dl = get_reasoning_dataloader(num_samples=200, batch_size=16, seed=seed)
    ntp_iter = iter(ntp_dl)

    records = []

    # Pre-train for 200 steps, snapshotting checkpoints every 40 steps
    for step in range(1, 201):
        try:
            batch = next(ntp_iter)
        except StopIteration:
            ntp_iter = iter(ntp_dl)
            batch = next(ntp_iter)

        train_ntp_step(model, optimizer, batch, device)

        if step % 40 == 0:
            print(f"Extracting readiness signals at Step {step}...")
            eval_ntp_batch = next(iter(ntp_dl))
            eval_rl_batch = next(iter(reason_dl))

            # 1. Extract Pre-RL signals
            signals = extract_checkpoint_signals(model, eval_ntp_batch, eval_rl_batch, verifier, device, step)

            # Measure baseline pass@1 pre-RL
            pre_eval = evaluate_model_capabilities(model, reason_dl, verifier, device)
            pre_pass1 = pre_eval["pass_at_1"]

            # 2. Run RL Excursion (30 steps of GRPO) from a deep copy of model
            excursion_model = LightweightLM(vocab_size=1000, d_model=64, n_layer=4, n_head=4, max_seq_len=64)
            excursion_model.load_state_dict(model.state_dict())
            excursion_opt = torch.optim.AdamW(excursion_model.parameters(), lr=5e-4)

            reason_iter = iter(reason_dl)
            for rl_step in range(30):
                try:
                    rl_b = next(reason_iter)
                except StopIteration:
                    reason_iter = iter(reason_dl)
                    rl_b = next(reason_iter)
                train_grpo_step(excursion_model, excursion_opt, rl_b, verifier, device, group_size=4)

            # Measure post-RL performance
            post_eval = evaluate_model_capabilities(excursion_model, reason_dl, verifier, device)
            post_pass1 = post_eval["pass_at_1"]
            rl_gain = post_pass1 - pre_pass1

            # Categorize Plasticity Response
            if rl_gain > 0.15:
                category = "EXPANSION"
            elif rl_gain > 0.02:
                category = "SHARPENING"
            elif rl_gain >= -0.02:
                category = "STAGNATION"
            else:
                category = "INTERFERENCE"

            rec = {
                "checkpoint_step": step,
                "gradient_alignment": signals["gradient_alignment"],
                "policy_entropy": signals["policy_entropy"],
                "pre_pass_at_k": signals["pass_at_k"],
                "pre_pass_at_1": pre_pass1,
                "post_pass_at_1": post_pass1,
                "rl_gain": rl_gain,
                "category": category,
            }
            records.append(rec)
            print(f"Step {step}: GradAlign={rec['gradient_alignment']:.4f}, Entropy={rec['policy_entropy']:.4f}, PrePass1={pre_pass1:.3f}, PostPass1={post_pass1:.3f}, RLGain={rl_gain:.3f} [{category}]")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved plasticity dataset with {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="artifacts/plasticity/checkpoint_rl_outcomes.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_plasticity_study(args.output, args.seed)
