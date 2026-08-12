# Null Results & Unstable Signal Log

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  

---

## Retained Negative & Null Findings

1. **Unstable Signal: Micro-Step Loss Variance**
   - *Finding:* Single-step NTP loss variance across mini-batches yielded noisy, uninformative correlations with subsequent RL gain ($r = 0.042, p = 0.74$).
   - *Resolution:* Replaced micro-step loss variance with macro policy entropy $H(\pi_\theta)$ evaluated over prompt batches.

2. **Negative Signal: KL Divergence Drift**
   - *Finding:* KL divergence drift from base policy exhibits a strong negative correlation with RL gain ($r = -0.6620, p = 8.42 \times 10^{-9}$). Large KL divergence early in training signals policy instability rather than constructive adaptation.

3. **Failed Baseline: Early Unconstrained RL (B3)**
   - *Finding:* Applying aggressive RL excursions at very early pre-training steps (before step 30) without SFT or NTP baseline anchoring resulted in policy entropy collapse and poor downstream Pass@1 ($35.74\%$).

4. **Non-Predictive Feature: Raw Token Count / Training Step Number**
   - *Finding:* Training step number alone produces a non-linear, non-monotonic relationship with RL gain ($R^2 = 0.124$). Standardizing step number fails to predict RL gain across models with different context lengths or vocabulary sizes.
