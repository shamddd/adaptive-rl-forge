# Novelty & Scientific Positioning Analysis

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  

---

## 1. What already exists?
- **Sequential Pipeline (NTP $\rightarrow$ SFT $\rightarrow$ RL):** Standard paradigm (e.g. InstructGPT, LLaMA-3, DeepSeek-R1) where RL is applied strictly post-pre-training.
- **Fixed Pre-training RL Excursions (Bansal et al., 2026):** Demonstrates that applying RL at fixed intermediate pre-training checkpoints expands distribution coverage. However, the timing of excursions is chosen manually or uniformly.
- **Static Reinforcement Pre-training (RPT, Huang et al., 2025):** Re-frames next-token prediction as RL over reasoning corpora using a fixed mixture of objective functions.
- **Gradient Projection & Alignment (PCGrad, Yu et al., 2020; LearnAlign, Zhang et al., 2025):** Projects conflicting task gradients at the step level or filters post-training dataset samples.

---

## 2. What is closest to CARLS?
- **Fixed Pre-training RL Excursions (Bansal et al., 2026):** Applies RL during pre-training, but uses static/fixed checkpoint selection without diagnostic readiness signals.
- **Curriculum Learning & Compute Allocation (RPT & CGPO):** Adjusts data difficulty or task weightings, but lacks dynamic feedback control driven by checkpoint plasticity indicators.

---

## 3. Genuinely Distinct Contributions of CARLS
1. **Diagnostic RL Plasticity Indicators:** Systematic measurement of pre-RL checkpoint signals (gradient alignment $\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy $H(\pi_\theta)$, pass@k slope, reward variance) to predict subsequent RL performance gain before spending compute on RL.
2. **Capability-Aware Dynamic Scheduling (CARLS):** An adaptive controller ($\alpha_{\text{NTP}}(t), \beta_{\text{SFT}}(t), \gamma_{\text{RL}}(t)$) that dynamically toggles or balances compute allocation based on real-time checkpoint readiness.
3. **Distribution Expansion vs. Sharpening Dynamics:** Characterization of how early dynamic RL excursions maintain language plasticity and solution strategy diversity compared to late post-training sharpening.

---

## 4. Overlap & Scientific Reframing
- **Overlap:** The observation that intermediate checkpoints can undergo RL excursions is already established by Bansal et al. (2026).
- **Reframing:** We do *not* claim to be the first to apply RL to non-final checkpoints. Instead, our scientific contribution is answering **WHEN** and **HOW** to automate RL interventions using measurable checkpoint signals, turning manual heuristic excursions into a principled dynamic optimization problem.

---

## 5. Smallest Defensible Novel Scientific Claim

> *"Intermediate language model checkpoints exhibit predictable variations in reinforcement learning plasticity. Measurable pre-RL state signals—specifically gradient alignment between next-token prediction and task reward gradients, policy entropy, and baseline pass@k—correlate with subsequent RL efficiency. The Capability-Aware Reinforcement Learning Scheduler (CARLS) leverages these signals to dynamically schedule RL compute, achieving superior compute-normalized performance and capability retention compared to fixed or sequential pipelines."*
