# Reproducibility Guide — AdaptiveRL-Forge

This guide describes how to reproduce the empirical results, plasticity study, baseline comparisons, statistical analysis, and JMLR paper PDF.

## 1. System & Environment Setup
- **Python Version:** 3.10+
- **Hardware Requirements:** CPU or Apple Silicon MPS / NVIDIA CUDA GPU. No expensive cluster compute required for principal small-scale reproduction.

### Installation
```bash
git clone https://github.com/shamddd/adaptive-rl-forge.git
cd adaptive-rl-forge
pip install -e .
```

---

## 2. Reproduction Commands

### Run Unit Tests
```bash
PYTHONPATH=. pytest tests/
```

### Reproduce Checkpoint Plasticity Study (Phase 4)
```bash
PYTHONPATH=. python scripts/run_plasticity_study.py --output artifacts/plasticity/checkpoint_rl_outcomes.parquet --seed 42
```

### Reproduce Full Multi-Seed Baseline Suite (B0–B7) & Figures (Phase 3, 7, 13)
```bash
PYTHONPATH=. python scripts/run_all_baselines.py
```

### Compile JMLR Paper PDF (Phase 15)
```bash
PYTHONPATH=. python scripts/compile_paper.py
```

### End-to-End One-Command Reproduction
```bash
make reproduce-main
```
