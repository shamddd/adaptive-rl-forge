# Novelty & Scientific Positioning Analysis (Second Pass Audit)

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  

---

## 1. Primary Research Question & Reframed Scientific Target

> **Core Research Question:** *"Can properties of a pretrained language-model checkpoint measured BEFORE reinforcement learning predict how strongly that checkpoint will respond to a fixed future RL intervention?"*

In this formulation, CARLS is the downstream dynamic controller application built upon discovering predictive readiness signals.

---

## 2. Literature Matrix & Overlap Analysis

| Prior Work | Scope & Measurements | Key Difference & Distinction of CARLS |
| :--- | :--- | :--- |
| **Bansal et al. (2026)** | Fixed intermediate RL excursions during pre-training | CARLS dynamically evaluates pre-RL diagnostic signals ($\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy) to automate intervention timing. |
| **Huang et al. (2025)** | Static Reinforcement Pre-Training (RPT) | CARLS dynamically adjusts objective weights rather than using fixed loss mixtures. |
| **Yu et al. (2020)** | Micro-level gradient projection (PCGrad) | CARLS uses gradient alignment as a macro checkpoint readiness indicator for compute allocation. |
| **Zhang et al. (2025)** | Post-training RL data selection (LearnAlign) | CARLS allocates compute across pre-training/SFT/RL phases rather than filtering post-training data. |

---

## 3. Defensible Novel Scientific Claims (Hypotheses under Validation)

1. **Predictability of RL Plasticity:** Intermediate language model checkpoints exhibit predictable variation in RL responsiveness, which can be forecasted prior to RL execution using diagnostic signals ($\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy, baseline accuracy).
2. **Dynamic Compute Allocation:** Signal-driven adaptive compute allocation between NTP, SFT, and RL can optimize compute efficiency compared to fixed-interval schedules.
