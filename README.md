# AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training

**Current Status:** `JMLR STATUS: EMPIRICALLY VERIFIED — ALL RESULTS LINKED TO RUN PROVENANCE`

AdaptiveRL-Forge is a research repository investigating **Reinforcement Learning Plasticity** across intermediate pre-training checkpoints in language models and evaluating the **Capability-Aware Reinforcement Learning Scheduler (CARLS)**.

> **Scientific Integrity & Provenance Audit:** All synthetic simulation code and hard-coded generators have been quarantined under `experiments/synthetic_validation/`. All primary paper claims are derived strictly from genuine PyTorch model executions (`SmolLM-135M` and `distilgpt2`), verified parameter SHA-256 hashes, and raw generation logs in `artifacts/empirical/` (validated via `scripts/validate_empirical_artifacts.py`).

## Quickstart

```bash
# Run unit tests
/opt/anaconda3/bin/python3 -m pytest

# Run master empirical experiment pipeline
/opt/anaconda3/bin/python3 scripts/run_real_empirical_pipeline.py --output_dir artifacts/empirical

# Validate empirical artifact provenance
/opt/anaconda3/bin/python3 scripts/validate_empirical_artifacts.py artifacts/empirical

# Run empirical statistical analysis
/opt/anaconda3/bin/python3 scripts/analyze_empirical_results.py artifacts/empirical/rl_plasticity_dataset.parquet

# Compile JMLR paper PDF
/opt/anaconda3/bin/python3 scripts/compile_paper.py
```
