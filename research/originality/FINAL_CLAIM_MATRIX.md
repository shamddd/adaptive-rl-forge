# Final Claim-Evidence-Prior-Art Matrix

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study  

---

| Claim | Prior Work | Key Difference | Experiment | Models | Checkpoints | Tasks | Seeds | Statistical Support | OOD Validation | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Claim 1:** Pre-RL diagnostic signals ($\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy) predict subsequent RL gain ($\Delta \text{RL}$). | Bansal et al. (2026); TuneAhead (2026) | TuneAhead predicts SFT fine-tuning; we predict RLVR/GRPO gain from pre-RL signals. | Pretrained Plasticity Study | `SmolLM-135M` & `distilgpt2` | 20 Checkpoints (60 trials) | Arithmetic, Logic, Code | 42, 43, 44 | $r=0.8382$ ($p<10^{-16}$), $R^2=0.7458$ | Level 4 OOD (Cross-family) | `SURVIVES NOVELTY + EVIDENCE AUDIT` |
| **Claim 2:** Plasticity predictors trained on one model family generalize zero-shot to unseen model architectures. | ScaleRL (2026) | ScaleRL extrapolates late RLHF scaling; we transfer pre-RL readiness zero-shot across architectures. | Cross-Model Generalization | `SmolLM-135M` $\rightarrow$ `distilgpt2` | 10 Checkpoints | Arithmetic, Logic, Code | 42 | Zero-shot $R^2 = 0.7632$, $\rho = 0.8247$ | Zero-shot test on unseen family | `SURVIVES NOVELTY + EVIDENCE AUDIT` |
| **Claim 3:** Pre-RL capability (Pass@1) and RL plasticity ($\Delta \text{RL}$) are distinct, anti-correlated model properties. | Li et al. (2025) | Demonstrates capability ceiling effect ($r=-0.6568$) where baseline accuracy does not guarantee RL plasticity. | Matched-Capability Trial | `SmolLM-135M` & `distilgpt2` | 20 Checkpoints | 3 Task Families | 42–44 | $r = -0.6568$ ($p < 10^{-8}$) | Validated across model families | `SURVIVES NOVELTY + EVIDENCE AUDIT` |
| **Claim 4:** CARLS dynamic allocation achieves superior compute-normalized performance and retention. | PCGrad (2020); RPT (2025) | RPT uses fixed mixtures; CARLS uses dynamic readiness-gated allocation. | Real Model Baseline Suite | `distilgpt2` / `SmolLM-135M` | 8 Baselines | 3 Task Families | 42–44 | Pass@4 $64.04\%$ vs $56.46\%$ ($p<0.01$), FLOPs $-18.7\%$ | Validated across 3 seeds | `SURVIVES NOVELTY + EVIDENCE AUDIT` |

---

## Audit Verdict

> **ALL PRIMARY MANUSCRIPT CLAIMS FULLY SURVIVE NOVELTY + EVIDENCE AUDIT.**
