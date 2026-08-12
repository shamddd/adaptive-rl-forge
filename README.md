# AdaptiveRL-Forge: Capability-Aware Dynamic Reinforcement Learning for Foundation Model Training

Official research repository for *When Should Language Models Learn from Reinforcement? Predicting Plasticity and Adaptively Scheduling Reinforcement Learning During Foundation-Model Training*.

## Quickstart
```bash
# Run unit tests
PYTHONPATH=. pytest tests/

# Execute baseline & CARLS experiments
PYTHONPATH=. python scripts/run_experiment.py --baseline B7_CARLS

# Compile JMLR paper
cd paper/jmlr && pdflatex main.tex
```
