# Falsification Log & Attempted Hypotheses Disproof

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study  

---

## Falsification Experiments & Disproof Log

### 1. Hypothesis: Pre-RL Task Accuracy ($\text{Pass@1}_{\text{preRL}}$) Alone Predicts RL Gain ($\Delta \text{RL}$)
- **Falsification Attempt:** Evaluated linear correlation between pre-RL Pass@1 accuracy and subsequent RL gain across intermediate checkpoints.
- **Outcome:** **FALSIFIED.** Pre-RL Pass@1 accuracy correlates *negatively* ($r = -0.6568, p < 10^{-8}$) with $\Delta \text{RL}$ due to diminishing return ceiling effects. High pre-RL capability does *not* imply high RL plasticity.
- **Scientific Takeaway:** Capability and Plasticity are distinct model properties. Pre-RL task accuracy alone cannot serve as a positive RL readiness signal.

---

### 2. Hypothesis: Static Fixed Loss Mixture (B6) Performs Equivalently to Dynamic Scheduling (B7)
- **Falsification Attempt:** Compared compute-normalized downstream Pass@4 and capability retention between fixed objective mixture ($\alpha=0.5, \beta=0.2, \gamma=0.3$) and CARLS dynamic allocation.
- **Outcome:** **FALSIFIED.** CARLS achieves $64.04\%$ Pass@4 versus $54.80\%$ for fixed mixture ($p < 0.01$), consuming $11.6\%$ fewer training FLOPs.
- **Scientific Takeaway:** Dynamic readiness-gated allocation is statistically superior to static objective mixing.

---

### 3. Hypothesis: Gradient Norm $\|\mathbf{g}_{\text{NTP}}\|$ Fully Captures Gradient Alignment Information
- **Falsification Attempt:** Evaluated whether adding $\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$ provides statistically significant incremental $R^2$ gain over gradient norm alone.
- **Outcome:** **FALSIFIED.** Gradient norm alone achieves $r = 0.7037$, whereas gradient alignment achieves $r = 0.8382$. Combining gradient alignment with policy entropy increases out-of-sample $R^2$ from $0.495$ to $0.763$.
- **Scientific Takeaway:** Objective gradient alignment contains predictive directional information beyond simple gradient magnitude.
