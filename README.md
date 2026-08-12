# AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training

**Current Status:** `JMLR STATUS: NOT READY — LARGE-SCALE EMPIRICAL BLOCKERS REMAIN`

AdaptiveRL-Forge is a research repository investigating **Reinforcement Learning Plasticity** across intermediate pre-training checkpoints in language models and evaluating the **Capability-Aware Reinforcement Learning Scheduler (CARLS)**.

> **Note on Pilot Phase:** Initial pilot experiments (68K–268K parameters) validate code paths and pipeline functionality. Comprehensive empirical validation on 100M–1.5B+ open language models (e.g., SmolLM, Qwen) is currently underway.

## Quickstart
```bash
# Run unit tests
PYTHONPATH=. pytest tests/

# Execute pilot baseline suite
PYTHONPATH=. python scripts/run_all_baselines.py

# Compile JMLR paper PDF
PYTHONPATH=. python scripts/compile_paper.py
```
