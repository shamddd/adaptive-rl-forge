import torch
import pytest
from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.datasets.pretraining_corpus import get_ntp_dataloader
from adaptive_rl_forge.datasets.reasoning_benchmarks import get_reasoning_dataloader
from adaptive_rl_forge.rewards.verifiers import ExactMatchRewardVerifier
from adaptive_rl_forge.ntp.trainer import train_ntp_step
from adaptive_rl_forge.sft.trainer import train_sft_step
from adaptive_rl_forge.rl.grpo_trainer import train_grpo_step


def test_ntp_and_sft_and_grpo_steps():
    device = torch.device("cpu")
    model = LightweightLM(vocab_size=1000, d_model=32, n_layer=2, n_head=2, max_seq_len=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    verifier = ExactMatchRewardVerifier()

    ntp_dl = get_ntp_dataloader(num_samples=16, seq_len=32, vocab_size=1000, batch_size=4)
    reason_dl = get_reasoning_dataloader(num_samples=16, batch_size=4)

    ntp_batch = next(iter(ntp_dl))
    reason_batch = next(iter(reason_dl))

    # Test NTP step
    ntp_loss, ntp_time, ntp_toks = train_ntp_step(model, optimizer, ntp_batch, device)
    assert ntp_loss > 0.0
    assert ntp_toks > 0

    # Test SFT step
    sft_loss, sft_time, sft_toks = train_sft_step(model, optimizer, reason_batch, device)
    assert sft_loss > 0.0
    assert sft_toks > 0

    # Test GRPO step
    pg_loss, reward, kl, rl_toks, rl_time = train_grpo_step(model, optimizer, reason_batch, verifier, device, group_size=2)
    assert isinstance(pg_loss, float)
    assert rl_toks > 0
