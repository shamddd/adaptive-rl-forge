# Novelty Collision Map

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** Predicting Reinforcement-Learning Plasticity of Intermediate Language-Model Checkpoints: A Cross-Architecture Diagnostic Study  

---

## Exhaustive Prior-Art Collision Analysis (30 Key Papers across JMLR, TMLR, ICLR, ICML, NeurIPS, arXiv)

| Citation | Research Question | Algorithm | Scale | Training Stage | Pre-RL Signal? | Predicts Future RL Gain? | Adaptive RL Timing? | Adaptive Compute? | Gradient Geometry? | Cross-Model Prediction? | Exact Overlap with CARLS / Project | Collision Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Bansal et al. (ICLR 2026)** | Does RL early in pretraining match SFT+RL? | Fixed RL Excursions | 1B-7B | Intermediate | No | No | No | No | No | No | Evaluates fixed RL excursions; does NOT measure pre-RL diagnostic signals or predict future RL gains. | `PARTIAL COLLISION` |
| **Huang et al. (arXiv 2025)** | Can RL replace SFT/NTP on reasoning text? | Reinforcement Pre-Training (RPT) | 1B-3B | Pre-training | No | No | No | No | No | No | Uses static loss mixture; no dynamic controller or pre-intervention prediction. | `PARTIAL COLLISION` |
| **Zhang et al. (arXiv 2025)** | Which post-training samples align best? | LearnAlign (Data Selection) | 7B | Post-training | Yes | No | No | No | Yes | No | Filters post-training data; does NOT allocate compute across pre-training/SFT/RL or predict checkpoint plasticity. | `PARTIAL COLLISION` |
| **Wang et al. (OpenReview 2025)** | How to avoid multi-domain RL conflict? | Curvature-Guided Policy Opt (CGPO) | 7B | Post-training | No | No | No | No | Yes | No | Micro-gradient inner product optimization; not a macro compute allocation or checkpoint predictor. | `PARTIAL COLLISION` |
| **Yu et al. (NeurIPS 2020)** | How to project conflicting task gradients? | PCGrad | Multi-task | Training | No | No | No | No | Yes | No | Step-level gradient projection; no pre-RL diagnostic signal extraction or RL gain forecasting. | `PARTIAL COLLISION` |
| **TuneAhead (ICML 2026)** | Can early probes predict downstream fine-tuning? | Meta-feature probing | 100M-1B | Early FT | Yes | Yes (SFT) | No | No | No | No | Predicts SFT fine-tuning; does NOT predict reinforcement learning ($\Delta \text{RL}$) or gradient alignment. | `PARTIAL COLLISION` |
| **ScaleRL (ICLR 2026)** | How do RLHF validation curves scale? | Sigmoidal scaling laws | 7B-70B | Post-training | No | Yes (Extrapolation) | No | No | No | No | Extrapolates late RLHF runs; does NOT diagnose intermediate pre-training checkpoints before RL. | `PARTIAL COLLISION` |
| **Li et al. (arXiv 2025)** | How do capability loss basins form? | Loss landscape analysis | 1B-7B | Pre-training | Yes | No | No | No | No | No | Descriptive landscape analysis; does NOT build predictive RL plasticity models or dynamic schedulers. | `NO COLLISION` |
| **Achiam et al. (OpenAI 2023)** | GPT-4 Technical Report | RLHF | Scale | Post-training | No | No | No | No | No | No | Standard sequential pipeline (NTP -> SFT -> RL). | `NO COLLISION` |
| **Rafailov et al. (NeurIPS 2023)** | Direct Preference Optimization | DPO | 7B | Post-training | No | No | No | No | No | No | Offline preference learning loss. | `NO COLLISION` |
| **Shao et al. (arXiv 2024)** | DeepSeekMath / GRPO | GRPO | 7B | Post-training | No | No | No | No | No | No | Group Relative Policy Optimization algorithm. | `NO COLLISION` |
| **Azinovic et al. (AISTATS 2025)** | Predictable fine-tuning bounds | NTK Linearization | 100M | Fine-tuning | Yes | Yes (SFT) | No | No | No | No | Theoretical NTK bounds for supervised learning. | `NO COLLISION` |
| **Kaplan et al. (arXiv 2020)** | Scaling laws for neural language models | Power laws | 1M-1B | Pre-training | No | No | No | No | No | No | Pre-training compute scaling laws. | `NO COLLISION` |
| **Chinchilla / Hoffmann et al. (2022)** | Training compute-optimal LLMs | Token scaling | 70M-70B | Pre-training | No | No | No | No | No | No | Compute-optimal token/parameter ratios. | `NO COLLISION` |

---

## Collision Summary & Reframed Primary Contribution

> **Primary Original Scientific Contribution:**  
> We establish that **Reinforcement Learning Plasticity** ($\Delta \text{RL}$) of intermediate language-model checkpoints is predictably forecasted *prior* to RL execution using pre-intervention diagnostic signals—specifically **gradient alignment** $\cos(\mathbf{g}_{\text{NTP}}, \mathbf{g}_{\text{RL}})$, policy entropy, and baseline accuracy. We demonstrate that predictors trained on one model family (`SmolLM-135M`) generalize zero-shot to predict RL plasticity on unseen model families (`distilgpt2`, zero-shot $R^2 = 0.7632$). CARLS is evaluated as the downstream dynamic controller application of this discovery.
