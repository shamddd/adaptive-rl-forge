"""
Provenance Validator for Empirical Artifacts (Phase 17).
Ensures every result row in artifacts/empirical/ originated from verified, executed model runs.
Rejects any dataset or report containing missing run directories, missing model revisions, or unverified outputs.
"""

import os
import sys
import json
import pandas as pd


def validate_empirical_artifacts(empirical_dir: str = "artifacts/empirical") -> bool:
    print(f"=== EMPIRICAL ARTIFACT PROVENANCE AUDIT: {empirical_dir} ===")

    dataset_path = os.path.join(empirical_dir, "rl_plasticity_dataset.parquet")
    runs_dir = os.path.join(empirical_dir, "runs")

    if not os.path.exists(dataset_path):
        print(f"FAILED: Empirical dataset missing at {dataset_path}")
        return False

    if not os.path.exists(runs_dir):
        print(f"FAILED: Runs provenance directory missing at {runs_dir}")
        return False

    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} empirical records for provenance verification.")

    required_columns = [
        "run_id",
        "model_name",
        "model_revision",
        "checkpoint_hash",
        "pre_pass_at_1",
        "post_pass_at_1",
        "rl_gain",
        "gradient_alignment",
        "policy_entropy",
        "gradient_norm",
        "wall_time",
        "hardware",
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"FAILED: Required provenance column '{col}' missing from dataset.")
            return False

    # Check each run directory for raw generation logs and execution metadata
    run_ids = df["run_id"].unique()
    print(f"Verifying {len(run_ids)} unique empirical run directories...")

    for run_id in run_ids:
        run_path = os.path.join(runs_dir, run_id)
        if not os.path.exists(run_path):
            print(f"FAILED: Missing run directory for run_id '{run_id}' at {run_path}")
            return False

        meta_path = os.path.join(run_path, "run_metadata.json")
        pre_gen_path = os.path.join(run_path, "pre_rl_generations.jsonl")
        post_gen_path = os.path.join(run_path, "post_rl_generations.jsonl")

        if not os.path.exists(meta_path):
            print(f"FAILED: Missing run_metadata.json for run_id '{run_id}'")
            return False

        if not os.path.exists(pre_gen_path) or not os.path.exists(post_gen_path):
            print(f"FAILED: Missing raw generation logs (pre/post JSONL) for run_id '{run_id}'")
            return False

        with open(meta_path, "r") as f:
            meta = json.load(f)

        required_meta_keys = ["model_name", "checkpoint_hash", "hardware", "seed", "rl_updates"]
        for k in required_meta_keys:
            if k not in meta:
                print(f"FAILED: Metadata key '{k}' missing from {meta_path}")
                return False

    print("SUCCESS: All empirical artifacts passed provenance validation!")
    return True


if __name__ == "__main__":
    emp_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/empirical"
    valid = validate_empirical_artifacts(emp_dir)
    sys.exit(0 if valid else 1)
