# Synthetic Validation Code & Artifacts

> **CRITICAL SCIENTIFIC INTEGRITY NOTICE**  
> The scripts and datasets in this directory are **development simulations and pipeline validation tests ONLY**.  
> They contain hard-coded formulas, simulated diagnostic signals, and synthetic noise distributions (`np.random.normal`).  
> **THEY DO NOT CONSTITUTE EMPIRICAL SCIENTIFIC EVIDENCE AND MUST NOT BE USED FOR PUBLICATION OR MANUSCRIPT CLAIMS.**  

## Directory Contents

- `synthetic_full_plasticity_demo.py`: Pipeline validation demo for plasticity regression routines.
- `synthetic_pretrained_plasticity_demo.py`: Pipeline validation demo for single-model signal extraction.
- `synthetic_baselines_demo.py`: Pipeline validation demo for baseline table rendering.
- `synthetic_experiment_demo.py`: Single-run pipeline test script.

All genuine empirical results are stored strictly under `artifacts/empirical/` and validated via `scripts/validate_empirical_artifacts.py`.
