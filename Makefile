.PHONY: test plasticity baselines paper reproduce-main clean

test:
	PYTHONPATH=. pytest tests/

plasticity:
	PYTHONPATH=. python scripts/run_plasticity_study.py --seed 42

baselines:
	PYTHONPATH=. python scripts/run_all_baselines.py

paper:
	PYTHONPATH=. python scripts/compile_paper.py

reproduce-main: test plasticity baselines paper
	@echo "=== Full Reproduction Complete! JMLR paper generated at paper/jmlr/main.pdf ==="

clean:
	rm -rf artifacts/runs/* artifacts/figures/* paper/jmlr/main.pdf
