# JMLR Final Publication Readiness Gate

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  
**Date:** August 12, 2026  

---

## Readiness Checklist & Verification Status

| Gate Criterion | Status | Empirical / Document Verification Path |
| :--- | :---: | :--- |
| **1. Software Architecture & Pipelines** | `PASS` | PyTorch/PEFT package `adaptive_rl_forge`; `pytest tests/` 100% pass |
| **2. Reproducibility Infrastructure** | `PASS` | `REPRODUCIBILITY.md` and `make reproduce-main` verified |
| **3. Pilot Baseline Implementations** | `PASS` | Baselines B0–B7 implemented with compute tracking |
| **4. Truthful Author Metadata** | `PASS` | Independent Researcher metadata in `main.tex` & `cover_letter.tex` |
| **5. Pretrained Model Scale Validation** | `BLOCKER` | Current experiments are 68K/268K pilot scale; 100M–1.5B+ models required |
| **6. Cross-Model Generalization** | `BLOCKER` | Plasticity predictors must be evaluated across $\ge 2$ real model families |
| **7. RL Plasticity Prediction Dataset** | `BLOCKER` | Need $\ge 8-12$ checkpoints per model family in `rl_plasticity_dataset.parquet` |
| **8. Multi-Task Family Benchmarking** | `BLOCKER` | Benchmark suite across arithmetic, logic, and code deterministic verifiers |
| **9. CARLS Superiority Validation** | `BLOCKER` | CARLS performance must be validated against SFT and sequential baselines on real LMs |
| **10. Claim-Evidence Alignment** | `BLOCKER` | All claims in manuscript must be fully supported by `CLAIM_EVIDENCE_MATRIX.md` |

---

## Final Submission Gate Status

> **JMLR STATUS: NOT READY — LARGE-SCALE EMPIRICAL BLOCKERS REMAIN**  
> The repository contains a complete software architecture, reproducibility workflow, and pilot experiment framework. However, it is **NOT** ready for JMLR submission until large-scale empirical validation on open pretrained language models (100M–1.5B+ parameters), cross-model generalization, and predictive plasticity evaluation are completed.
