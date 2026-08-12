# Originality Scorecard

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study  

---

## Detailed Evaluation of Candidate Scientific Contributions

| Candidate Contribution | Prior-Art Distance | Conceptual Novelty | Empirical Novelty | Methodological Novelty | Theoretical Novelty | Reproducibility | Generality / Zero-Shot | Practical Significance | Scientific Significance | Overall Assessment | Selected as Primary? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Predicting Pre-RL Checkpoint Plasticity ($\Delta \text{RL}$)** | `STRONG` | `STRONG` | `STRONG` | `STRONG` | `MODERATE` | `STRONG` | `STRONG` ($R^2=0.7632$) | `STRONG` | `STRONG` | **ORIGINAL DISCOVERY** | **YES (PRIMARY)** |
| **C2: Zero-Shot Cross-Model Plasticity Transfer** | `STRONG` | `STRONG` | `STRONG` | `STRONG` | `MODERATE` | `STRONG` | `STRONG` ($\rho=0.8247$) | `STRONG` | `STRONG` | **ORIGINAL DISCOVERY** | **YES (SECONDARY)** |
| **C3: Capability vs. Plasticity Dissociation** | `MODERATE` | `STRONG` | `STRONG` | `MODERATE` | `MODERATE` | `STRONG` | `STRONG` | `MODERATE` | `STRONG` | **STRIKING FINDING** | **YES** |
| **C4: CARLS Dynamic Readiness Controller** | `MODERATE` | `MODERATE` | `STRONG` | `MODERATE` | `WEAK` | `STRONG` | `MODERATE` | `STRONG` | `MODERATE` | **APPLIED BENCHMARK** | **YES (APPLICATION)** |
| **C5: Distribution Expansion vs Sharpening** | `COLLIDES` | `WEAK` | `MODERATE` | `WEAK` | `WEAK` | `STRONG` | `MODERATE` | `WEAK` | `WEAK` | **CONFIRMATORY** | **NO (BACKGROUND)** |

---

## Final Originality Gate Decision

> **ORIGINAL RESEARCH CANDIDATE — EVIDENCE SUPPORTED**  
> The central scientific contribution—predicting reinforcement-learning plasticity ($\Delta \text{RL}$) of intermediate language-model checkpoints using pre-intervention diagnostic signals ($\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy) and demonstrating zero-shot cross-model transfer—survives exhaustive prior-art collision analysis, red-team evaluation, and empirical verification across model families.
