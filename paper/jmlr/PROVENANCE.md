# Figure and Table Traceability Matrix

**Project Title:** AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training  
**Working Paper Title:** When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training  

---

## 1. Table Provenance

### Table 1: Compute-Normalized Performance & Solution Diversity
- **Source Script:** `scripts/run_all_baselines.py`
- **Output Artifact:** `artifacts/baseline_summary_table.csv`
- **Raw Run Inputs:** `artifacts/runs/B0_NTP_seed*/summary.json` through `artifacts/runs/B7_CARLS_seed*/summary.json`
- **Git Commit:** Head commit of repository `shamddd/adaptive-rl-forge`
- **Execution Log:** `artifacts/master_experiment_results.csv`

---

## 2. Figure Provenance

### Figure 1: Compute-Normalized Benchmark Accuracy (`figures/fig1_baseline_accuracy.png`)
- **Source Script:** `scripts/run_all_baselines.py`
- **Input Data:** `artifacts/baseline_summary_table.csv`
- **Generated File Path:** `artifacts/figures/fig1_baseline_accuracy.png` $\rightarrow$ `paper/jmlr/figures/fig1_baseline_accuracy.png`

### Figure 2: Compute Efficiency Pareto Frontier (`figures/fig2_compute_pareto.png`)
- **Source Script:** `scripts/run_all_baselines.py`
- **Input Data:** `artifacts/baseline_summary_table.csv`
- **Generated File Path:** `artifacts/figures/fig2_compute_pareto.png` $\rightarrow$ `paper/jmlr/figures/fig2_compute_pareto.png`

### Checkpoint Plasticity Outcome Dataset (`artifacts/plasticity/checkpoint_rl_outcomes.parquet`)
- **Source Script:** `scripts/run_plasticity_study.py`
- **Input Seed:** 42
