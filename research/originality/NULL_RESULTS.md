# Null Results & Unstable Signal Log

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  

---

## Retained Negative & Null Findings

1. **Unstable Signal: Micro-Step Loss Variance**
   - *Finding:* Single-step NTP loss variance across mini-batches yielded noisy, uninformative correlations with subsequent RL gain ($r = 0.042, p = 0.74$).
   - *Resolution:* Replaced micro-step loss variance with macro policy entropy $H(\pi_\theta)$ evaluated over prompt batches.

2. **Negative Signal: KL Divergence Penalty**
   - *Finding:* KL divergence drift from base policy exhibits a strong negative correlation with RL gain ($r = -0.6620, p < 10^{-8}$). Large KL divergence early in training signals policy instability rather than constructive adaptation.

3. **Failed Baseline: Early Unconstrained RL (B3)**
   - *Finding:* Applying aggressive RL excursions at very early pre-training steps (before step 30) without SFT or NTP baseline anchoring resulted in policy entropy collapse and poor downstream Pass@1 ($35.01\%$).
