import torch
import pytest
from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.datasets.pretraining_corpus import get_ntp_dataloader
from adaptive_rl_forge.datasets.reasoning_benchmarks import get_reasoning_dataloader
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier
from adaptive_rl_forge.carls.signals import extract_checkpoint_signals
from adaptive_rl_forge.carls.controller import CARLSv0Controller, CARLSv1Controller


def test_carls_signals_and_controllers():
    device = torch.device("cpu")
    model = LightweightLM(vocab_size=1000, d_model=32, n_layer=2, n_head=2, max_seq_len=64)
    verifier = ExactMatchRewardVerifier()

    ntp_batch = next(iter(get_ntp_dataloader(num_samples=4, seq_len=32, vocab_size=1000, batch_size=4)))
    rl_batch = next(iter(get_reasoning_dataloader(num_samples=4, batch_size=4)))

    signals = extract_checkpoint_signals(model, ntp_batch, rl_batch, verifier, device, step=10)
    assert "gradient_alignment" in signals
    assert "policy_entropy" in signals
    assert "pass_at_k" in signals

    # Test CARLS v0 controller
    ctrl_v0 = CARLSv0Controller()
    a, b, c = ctrl_v0.compute_allocation(signals)
    assert pytest.approx(a + b + c) == 1.0

    # Test CARLS v1 controller
    ctrl_v1 = CARLSv1Controller()
    a1, b1, c1 = ctrl_v1.compute_allocation(signals)
    assert pytest.approx(a1 + b1 + c1) == 1.0
