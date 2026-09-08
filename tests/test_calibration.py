"""
Unit tests for calibration evaluation metrics and calibration-aware GRPO step.
"""

import pytest
import torch
from adaptive_rl_forge.eval.evaluator import compute_expected_calibration_error
from adaptive_rl_forge.rl.grpo_trainer import train_grpo_step
from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier


def test_ece_computation_perfect_calibration():
    # Perfectly calibrated predictions: conf=1.0, correct=1; conf=0.0, correct=0
    confidences = [1.0, 1.0, 0.0, 0.0]
    correctness = [1, 1, 0, 0]
    metrics = compute_expected_calibration_error(confidences, correctness, num_bins=5)
    assert abs(metrics["ece"]) < 1e-5
    assert abs(metrics["brier_score"]) < 1e-5


def test_ece_computation_miscalibration():
    # Completely miscalibrated predictions: conf=1.0 when wrong (0), conf=0.0 when right (1)
    confidences = [0.95, 0.95, 0.05, 0.05]
    correctness = [0, 0, 1, 1]
    metrics = compute_expected_calibration_error(confidences, correctness, num_bins=10)
    assert metrics["ece"] > 0.80
    assert metrics["brier_score"] > 0.80


def test_train_grpo_step_with_calibration_loss():
    torch.manual_seed(42)
    vocab_size = 128
    d_model = 32
    n_layer = 2

    model = LightweightLM(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    verifier = ExactMatchRewardVerifier()


    prompt_ids = torch.randint(0, vocab_size, (2, 4))
    answer_tokens = torch.randint(0, vocab_size, (2,))
    batch = {"prompt_ids": prompt_ids, "answer_tokens": answer_tokens}

    device = torch.device("cpu")
    pg_loss, mean_r, kl_loss, num_tokens, elapsed = train_grpo_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        verifier=verifier,
        device=device,
        group_size=2,
        kl_coeff=0.0,
        calibration_coeff=0.1,
    )

    assert isinstance(pg_loss, float)
    assert isinstance(mean_r, float)
    assert num_tokens > 0
    assert elapsed > 0.0
