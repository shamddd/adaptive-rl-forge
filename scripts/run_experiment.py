"""
Main Experiment Runner Script for Baselines B0–B7 and CARLS (Phase 3 & Phase 7–13).
"""

import os
import sys
import json
import time
import argparse
import random
import torch
import numpy as np

from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.datasets.pretraining_corpus import get_ntp_dataloader
from adaptive_rl_forge.datasets.reasoning_benchmarks import get_reasoning_dataloader
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier
from adaptive_rl_forge.ntp.trainer import train_ntp_step
from adaptive_rl_forge.sft.trainer import train_sft_step
from adaptive_rl_forge.rl.grpo_trainer import train_grpo_step
from adaptive_rl_forge.carls.signals import extract_checkpoint_signals
from adaptive_rl_forge.carls.controller import CARLSv0Controller, CARLSv1Controller
from adaptive_rl_forge.metrics.tracker import ComputeTracker
from adaptive_rl_forge.eval.evaluator import evaluate_model_capabilities


def run_experiment(
    baseline: str,
    total_steps: int = 150,
    seed: int = 42,
    output_dir: str = "artifacts/runs",
    model_scale: str = "default",
):
    print(f"==================================================")
    print(f"Running Experiment: {baseline} | Seed: {seed} | Steps: {total_steps} | Model: {model_scale}")
    print(f"==================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")

    run_id = f"{baseline}_seed{seed}_{int(time.time())}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Model Configuration
    if model_scale == "tiny":
        model = LightweightLM(vocab_size=1000, d_model=32, n_layer=2, n_head=2, max_seq_len=64)
        ref_model = LightweightLM(vocab_size=1000, d_model=32, n_layer=2, n_head=2, max_seq_len=64)
    else:
        model = LightweightLM(vocab_size=1000, d_model=64, n_layer=4, n_head=4, max_seq_len=64)
        ref_model = LightweightLM(vocab_size=1000, d_model=64, n_layer=4, n_head=4, max_seq_len=64)

    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    tracker = ComputeTracker(num_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    verifier = ExactMatchRewardVerifier()

    ntp_dl = get_ntp_dataloader(num_samples=1000, seq_len=32, vocab_size=1000, batch_size=16, seed=seed)
    reason_dl = get_reasoning_dataloader(num_samples=500, batch_size=16, seed=seed)

    ntp_iter = iter(ntp_dl)
    reason_iter = iter(reason_dl)

    carls_v0 = CARLSv0Controller()
    carls_v1 = CARLSv1Controller()

    schedule_log = []

    for step in range(1, total_steps + 1):
        # Determine objective allocation according to Baseline
        if baseline == "B0_NTP":
            alpha, beta, gamma = 1.0, 0.0, 0.0
        elif baseline == "B1_SFT":
            alpha, beta, gamma = 0.0, 1.0, 0.0
        elif baseline == "B2_Sequential":
            if step <= total_steps * 0.6:
                alpha, beta, gamma = 1.0, 0.0, 0.0
            elif step <= total_steps * 0.8:
                alpha, beta, gamma = 0.0, 1.0, 0.0
            else:
                alpha, beta, gamma = 0.0, 0.0, 1.0
        elif baseline == "B3_EarlyRL":
            if 30 <= step <= 60:
                alpha, beta, gamma = 0.1, 0.1, 0.8
            else:
                alpha, beta, gamma = 0.8, 0.2, 0.0
        elif baseline == "B4_PeriodicRL":
            if step % 30 < 10:
                alpha, beta, gamma = 0.2, 0.1, 0.7
            else:
                alpha, beta, gamma = 0.8, 0.2, 0.0
        elif baseline == "B5_RandomRL":
            rng_val = random.random()
            if rng_val < 0.3:
                alpha, beta, gamma = 0.2, 0.1, 0.7
            else:
                alpha, beta, gamma = 0.7, 0.3, 0.0
        elif baseline == "B6_FixedMixture":
            alpha, beta, gamma = 0.5, 0.2, 0.3
        elif baseline == "B7_CARLS":
            if step % 10 == 1 or step == 1:
                eval_ntp_b = next(iter(ntp_dl))
                eval_rl_b = next(iter(reason_dl))
                signals = extract_checkpoint_signals(model, eval_ntp_b, eval_rl_b, verifier, device, step)
                alpha, beta, gamma = carls_v1.compute_allocation(signals)
            else:
                alpha, beta, gamma = schedule_log[-1]["alpha"], schedule_log[-1]["beta"], schedule_log[-1]["gamma"]
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        schedule_log.append({"step": step, "alpha": alpha, "beta": beta, "gamma": gamma})

        # Execute training step according to sampling
        obj_choice = random.choices(["NTP", "SFT", "RL"], weights=[alpha, beta, gamma])[0]

        if obj_choice == "NTP":
            try:
                b = next(ntp_iter)
            except StopIteration:
                ntp_iter = iter(ntp_dl)
                b = next(ntp_iter)
            loss, elapsed, toks = train_ntp_step(model, optimizer, b, device)
            tracker.record_step(step, "NTP", loss, toks, elapsed)
        elif obj_choice == "SFT":
            try:
                b = next(reason_iter)
            except StopIteration:
                reason_iter = iter(reason_dl)
                b = next(reason_iter)
            loss, elapsed, toks = train_sft_step(model, optimizer, b, device)
            tracker.record_step(step, "SFT", loss, toks, elapsed)
        else:  # RL / GRPO
            try:
                b = next(reason_iter)
            except StopIteration:
                reason_iter = iter(reason_dl)
                b = next(reason_iter)
            loss, r_mean, kl, toks, elapsed = train_grpo_step(model, optimizer, b, verifier, device, group_size=4, ref_model=ref_model)
            tracker.record_step(step, "RL", loss, toks, elapsed, {"reward": r_mean, "kl": kl})

        if step % 30 == 0 or step == total_steps:
            print(f"Step {step}/{total_steps} [{obj_choice}]: Loss={tracker.step_history[-1]['loss']:.4f}")

    # Evaluate final model capabilities
    print("Evaluating final model capabilities...")
    final_eval = evaluate_model_capabilities(model, reason_dl, verifier, device)
    summary_data = tracker.summary()
    summary_data.update(final_eval)

    # Save Provenance Artifacts
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(f"baseline: {baseline}\nseed: {seed}\ntotal_steps: {total_steps}\nmodel_num_params: {num_params}\nmodel_scale: {model_scale}\n")

    with open(os.path.join(run_dir, "metrics.jsonl"), "w") as f:
        for item in tracker.step_history:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(run_dir, "schedule_log.json"), "w") as f:
        json.dump(schedule_log, f, indent=2)

    print(f"Experiment {baseline} completed! Final Pass@1: {summary_data['pass_at_1']:.4f}, Pass@4: {summary_data['pass_at_4']:.4f}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="B7_CARLS")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts/runs")
    parser.add_argument("--model_scale", type=str, default="default")
    args = parser.parse_args()

    run_experiment(args.baseline, args.steps, args.seed, args.output_dir, args.model_scale)
