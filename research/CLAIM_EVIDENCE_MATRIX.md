# Claim-Evidence Traceability Matrix

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  

---

## Abstract & Manuscript Claim Audit

| Claim ID | Abstract / Section Claim | Supporting Evidence | Experiment IDs / Model / Seeds | Statistical Support | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **C1** | Pre-RL diagnostic signals predict subsequent RL plasticity gain. | Pretrained LM Plasticity Study | `distilgpt2` / Seed 42 / 8 Checkpoints | $R^2 = 0.7458$, Entropy $r=0.927$ ($p<0.001$), GradAlign $r=0.606$ | `SUPPORTED` |
| **C2** | Early RL excursions demonstrate output probability expansion. | Pilot Baseline Suite & Pretrained Plasticity | SmolLM / distilgpt2 / Seeds 42–44 | Higher Diversity Ratios ($0.032 - 0.035$) vs SFT ($0.015$) | `PARTIALLY SUPPORTED` (Pilot Scale) |
| **C3** | CARLS dynamic compute allocation achieves superior compute-normalized performance. | Pilot Baseline Suite | Baselines B0–B7 / Seeds 42–44 | CARLS consumes $6.20 \times 10^{10}$ FLOPs vs NTP $9.84 \times 10^{10}$ | `HYPOTHESIS / PILOT` (Requires 1B+ Scale Validation) |
| **C4** | Truthful author metadata without unverified institutional affiliations. | Manuscript Source Files | `main.tex`, `cover_letter.tex` | Independent Researcher metadata verified | `SUPPORTED` |

---

## Allowed Status Summary
- **SUPPORTED:** Empirical evidence from real/pretrained experiments supports the claim.
- **PARTIALLY SUPPORTED:** Pilot-scale or initial model evidence supports the claim; further scale validation ongoing.
- **HYPOTHESIS / PILOT:** Claim is framed strictly as a hypothesis or pilot observation pending full-scale multi-billion parameter validation.
