import os
import torch
import pytest
from adaptive_rl_forge.models.lightweight_lm import LightweightLM
from adaptive_rl_forge.models.checkpoint_tracker import CheckpointTracker


def test_lightweight_lm_forward_and_generate():
    model = LightweightLM(vocab_size=100, d_model=32, n_layer=2, n_head=2, max_seq_len=32)
    input_ids = torch.randint(0, 100, (2, 10))
    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (2, 10, 100)
    assert loss is not None
    assert loss.item() > 0.0

    # Generation test
    prompt = torch.randint(0, 100, (1, 5))
    gen = model.generate(prompt, max_new_tokens=5)
    assert gen.shape == (1, 10)


def test_checkpoint_tracker(tmp_path):
    save_dir = str(tmp_path / "checkpoints")
    tracker = CheckpointTracker(save_dir)
    model = LightweightLM(vocab_size=100, d_model=32, n_layer=2, n_head=2, max_seq_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = tracker.save_checkpoint(model, opt, step=10, metrics={"loss": 0.5})
    assert os.path.exists(path)

    new_model = LightweightLM(vocab_size=100, d_model=32, n_layer=2, n_head=2, max_seq_len=32)
    ckpt = tracker.load_checkpoint(path, new_model)
    assert ckpt["step"] == 10
    assert ckpt["metrics"]["loss"] == 0.5
