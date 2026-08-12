# JMLR Readiness Audit: AdaptiveRL-Forge

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  
**Date:** August 12, 2026  
**Auditor:** Autonomous Research Engineering Agent  

---

## Executive Summary Audit Table

| Category | Status | Notes & Required Actions |
| :--- | :---: | :--- |
| **1. Originality & Scope** | `IMPORTANT` | CARLS formulation is novel; literature matrix in Phase 1 will confirm precise boundaries. |
| **2. Algorithm Completeness** | `IMPORTANT` | Core CARLS-v0 (rule-based) & CARLS-v1 (learned controller) specified; implementation underway in Phase 6. |
| **3. Experimental Evidence** | `BLOCKER` | Initial empirical experiments on checkpoint plasticity and schedule comparison must be executed. |
| **4. Baselines Coverage** | `IMPORTANT` | B0–B7 baselines defined; need full empirical runs with compute-normalized tracking. |
| **5. Number of Seeds** | `IMPORTANT` | 3 random seeds required for statistical confidence intervals. |
| **6. Checkpoint Diversity** | `IMPORTANT` | Intermediate checkpoints across early, mid, and late training stages to be sampled. |
| **7. Model Diversity** | `IMPORTANT` | Evaluation planned across two small LM scale configurations. |
| **8. Task Diversity** | `IMPORTANT` | Evaluated across math reasoning (GSM8K style), logic (ARC style), and code/language syntax. |
| **9. Ablations** | `IMPORTANT` | Signal-level ablations (gradient alignment, entropy, reward variance) to be run. |
| **10. Statistical Validity** | `IMPORTANT` | Hypotheses tests, confidence intervals, effect sizes documented in Phase 14. |
| **11. Reproducibility** | `IMPORTANT` | `REPRODUCIBILITY.md` and `make reproduce-main` script to be built. |
| **12. Compute Accounting** | `IMPORTANT` | Token, FLOPs, GPU/CPU hours, wall-clock time, and memory metrics recorded per run. |
| **13. Limitations & Safety** | `IMPORTANT` | Explicit limitations section required in manuscript and audit. |
| **14. Paper Completeness** | `IMPORTANT` | Official `jmlr2e.sty` LaTeX template and sections to be generated. |
| **15. Reference Integrity** | `IMPORTANT` | References to be audited against scholarly databases; no hallucinated citations. |
| **16. Figure & Table Provenance** | `IMPORTANT` | `paper/jmlr/PROVENANCE.md` to map every table cell and plot back to raw run outputs. |

---

## Detailed Category Assessments

### 1. Originality & Scope
- **Current State:** Dynamic scheduling of RL excursions during intermediate pre-training based on checkpoint readiness signals (gradient alignment, entropy drift) is conceptually distinct from static post-training RLHF/GRPO.
- **Action Required:** Complete Phase 1 literature audit matrix to ensure no direct collision with recent literature.

### 2. Experimental Evidence & Baselines
- **Current State:** Repository infrastructure being initialized. No fake data will be included.
- **Action Required:** Run Phase 4 & Phase 7 multi-seed experiments on small LMs to populate empirical metrics.

### 3. JMLR Submission Gate Status
- **Current Status:** `IN PROGRESS / NOT YET READY FOR SUBMISSION`
- **Blocker:** Real experimental results must be completed and validated before manuscript finalization.
