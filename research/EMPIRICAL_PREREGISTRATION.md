# Empirical Research Preregistration

**Date of Preregistration:** August 13, 2026  
**Status:** PREREGISTERED BEFORE EMPIRICAL RUNS  

---

## 1. Primary Hypothesis & Scientific Objective
We test whether pre-intervention diagnostic signals—specifically objective gradient alignment $\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$ and policy entropy $H(\pi_\theta)$—measured on intermediate model parameter states contain predictive information about subsequent reinforcement learning capability gain ($\Delta \text{RL}$) above trivial baseline indicators (training step / token count and pre-RL task accuracy).

## 2. Target Models & Checkpoint States
- **Model Family A:** `HuggingFaceTB/SmolLM-135M` (135M parameters)
- **Model Family B:** `distilgpt2` (82M parameters)
- **Checkpoint Generation:** Controlled continued pretraining over Wikitext / OpenWebText corpus to produce 6 distinct parameter states per model family (12 total checkpoints). Every checkpoint is uniquely identified by parameter SHA-256 hash.

## 3. Pre-RL Diagnostic Measurements
- **NTP Loss:** $\mathcal{L}_{\text{NTP}}$ on held-out diagnostic text batch.
- **Policy Entropy:** $H(\pi_\theta) = -\sum_v \pi_\theta(v|x) \log \pi_\theta(v|x)$ over task prompts.
- **Gradient Alignment:** $\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}}) = \frac{\langle \mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}} \rangle}{\|\mathbf{g}_{\text{NTP}}\| \|\mathbf{g}_{\text{RL}}\|}$ evaluated on LoRA adapter parameters ($\theta_{\text{LoRA}}$).
- **Gradient Norm:** $\|\mathbf{g}_{\text{NTP}}\|$.
- **Baseline Accuracy:** Pre-RL Pass@1 and Pass@4 on evaluation tasks.
- **Pre-RL Retention:** WikiText language modeling perplexity.

## 4. Standardized RL Plasticity Probe ($R^*$)
Applied identically to every checkpoint:
- **Optimizer:** AdamW ($lr = 5 \times 10^{-5}$)
- **RL Algorithm:** Group Relative Policy Optimization (GRPO)
- **Group Size ($G$):** 4 rollouts per prompt
- **Prompt Budget:** 20 prompt batches per checkpoint
- **Update Steps:** 20 steps
- **KL Coefficient:** $\beta_{\text{KL}} = 0.05$
- **Generation Settings:** Temperature $0.7$, Max New Tokens $32$

## 5. Evaluation Tasks & Verification
- **Arithmetic:** Exact-match numerical verification on multi-digit addition/subtraction.
- **Logic:** Deterministic boolean truth-table verification.
- **Code:** Python snippet execution verification.

## 6. Primary Outcome Metric
$$\Delta \text{RL} = \text{Pass@1}_{\text{postRL}} - \text{Pass@1}_{\text{preRL}}$$

## 7. Statistical Tests & Predictor Evaluation
- **Predictor Comparison:** Compare linear models M0 (step only), M1 (pre-RL accuracy only), M2 (entropy only), M3 (gradient alignment only), M4 (full diagnostic model M0+M1+M2+M3).
- **Metric:** Out-of-sample $R^2$, Pearson $r$, Spearman $\rho$, paired t-test $p$-values, 95% bootstrap confidence intervals.
- **Cross-Model Transfer:** Train predictor on Model Family A (`SmolLM-135M`), evaluate zero-shot on untouched Model Family B (`distilgpt2`), and reverse B $\to$ A.

## 8. Success / Failure Criteria
- **Success:** Full diagnostic model (M4) achieves statistically significant out-of-sample prediction gain over trivial baseline (M0+M1) ($p < 0.05$), and zero-shot cross-model transfer $R^2 > 0.0$.
- **Failure:** If gradient alignment adds no out-of-sample predictive power over baseline accuracy, or zero-shot transfer fails ($R^2 \le 0.0$), the hypothesis is rejected.
