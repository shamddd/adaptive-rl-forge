# Statistical Analysis & Hypothesis Testing — Pretrained LM Suite

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  

---

## 1. Pretrained Model Experimental Setup
- **Pretrained Model Families:** `HuggingFaceTB/SmolLM-135M` (134.5M params) and `distilgpt2` (81.9M params) with PEFT LoRA adapters.
- **Intervention Trials:** 60 total checkpoint interventions (10 checkpoints $\times$ 2 model families $\times$ 3 task families).
- **Task Families:** Arithmetic Reasoning, Symbolic Logic, and Code Execution.
- **Random Seeds:** 3 distinct random seeds (42, 43, 44) per baseline schedule.
- **Uncertainty Estimation:** Non-parametric bootstrap resampling (1,000 resamples, 95% percentile confidence intervals).

---

## 2. Empirical Baseline Summary Table (Pretrained LMs)

| Baseline Schedule | Pass@1 Mean (%) | Pass@1 95% CI | Pass@4 Mean (%) | Pass@4 95% CI | Diversity Ratio | Retention Score | FLOPs ($\times 10^{10}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: NTP Only** | 4.57 | [3.50, 6.10] | 11.70 | [10.70, 12.60] | 0.0457 | 0.88 | 9.80 |
| **B1: SFT Only** | 40.58 | [38.20, 42.60] | 50.55 | [47.80, 54.20] | 0.0180 | 0.88 | 2.50 |
| **B2: Sequential Pipeline** | 42.49 | [41.00, 44.30] | 56.46 | [53.40, 58.60] | 0.0292 | 0.85 | 7.50 |
| **B3: Early RL Excursion** | 35.01 | [32.20, 37.40] | 47.38 | [45.70, 49.60] | 0.0494 | 0.88 | 7.60 |
| **B4: Periodic RL Excursions** | 40.96 | [39.50, 42.10] | 56.50 | [55.50, 57.30] | 0.0427 | 0.88 | 7.60 |
| **B5: Random RL Timing** | 38.84 | [38.40, 39.50] | 54.59 | [53.40, 56.40] | 0.0473 | 0.88 | 7.20 |
| **B6: Fixed Objective Mixture** | 38.56 | [36.60, 39.60] | 54.80 | [54.20, 55.80] | 0.0299 | 0.88 | 6.90 |
| **B7: CARLS (Ours)** | **47.26** | **[44.60, 50.00]** | **64.04** | **[62.80, 65.30]** | **0.0388** | **0.94** | **6.10** |

---

## 3. Pre-RL Diagnostic Signal Correlations & Cross-Model Generalization

| Pre-RL Diagnostic Signal | Pearson $r$ | $p$-value | Spearman $\rho$ | $p$-value | Predictive Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gradient Alignment ($\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$)** | **+0.8382** | $6.65 \times 10^{-17}$ | **+0.7754** | $3.49 \times 10^{-13}$ | **Strongest Positive Predictor** |
| **Policy Entropy ($H(\pi_\theta)$)** | **+0.7439** | $9.72 \times 10^{-12}$ | **+0.6533** | $1.53 \times 10^{-8}$ | **High Entropy Enables Plasticity** |
| **Gradient Norm ($\|\mathbf{g}_{\text{NTP}}\||$)** | **+0.7037** | $3.58 \times 10^{-10}$ | **+0.6395** | $3.81 \times 10^{-8}$ | Active Representation Learning |
| **Reward Variance ($\text{Var}(R)$)** | **+0.6702** | $4.70 \times 10^{-9}$ | **+0.5818** | $1.09 \times 10^{-6}$ | Policy Exploratory Signal |
| **Pre-RL Pass@1 Accuracy** | **-0.6568** | $1.20 \times 10^{-8}$ | **-0.6367** | $4.53 \times 10^{-8}$ | Ceiling Effect / Diminishing Gain |
| **KL Drift** | **-0.6620** | $8.42 \times 10^{-9}$ | **-0.6138** | $1.84 \times 10^{-7}$ | Policy Divergence Penalty |

### Zero-Shot Cross-Model Predictor Generalization
- **Training Model:** `HuggingFaceTB/SmolLM-135M` (In-domain fit $R^2 = 0.6976$)
- **Evaluation Model:** `distilgpt2` (Zero-shot test $R^2 = 0.7632$, Spearman $\rho = 0.8247$)
- **Conclusion:** Pre-RL readiness signals generalize zero-shot across distinct open model architectures.
