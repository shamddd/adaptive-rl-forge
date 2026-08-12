"""
Empirical Statistical Analysis & Hypothesis Testing (Phases 19-24).
Reads ONLY verified empirical records from artifacts/empirical/rl_plasticity_dataset.parquet.
Computes Pearson r, Spearman rho, R^2, and zero-shot cross-model transfer.
No synthetic metrics permitted.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge


def analyze_empirical_dataset(empirical_parquet: str = "artifacts/empirical/rl_plasticity_dataset.parquet"):
    print(f"=== EMPIRICAL STATISTICAL ANALYSIS: {empirical_parquet} ===")

    if not os.path.exists(empirical_parquet):
        print(f"ERROR: Empirical dataset missing at {empirical_parquet}")
        sys.exit(1)

    df = pd.read_parquet(empirical_parquet)
    print(f"Loaded {len(df)} verified empirical run records across seeds.")

    print("\n--- Summary Statistics of Empirical Run Variables ---")
    summary_cols = ["pre_pass_at_1", "post_pass_at_1", "rl_gain", "gradient_alignment", "policy_entropy", "retention_score"]
    print(df[summary_cols].describe().to_string())

    print("\n==========================================================================")
    print("EMPIRICAL CORRELATIONS WITH RL GAIN (Δ Pass@1)")
    print("==========================================================================")
    
    corr_results = []
    for feat in ["gradient_alignment", "policy_entropy", "pre_pass_at_1", "gradient_norm", "checkpoint_stage"]:
        r, p_val = stats.pearsonr(df[feat], df["rl_gain"])
        rho, rho_p = stats.spearmanr(df[feat], df["rl_gain"])
        corr_results.append({
            "feature": feat,
            "pearson_r": r,
            "pearson_p": p_val,
            "spearman_rho": rho,
            "spearman_p": rho_p,
        })

    corr_df = pd.DataFrame(corr_results)
    print(corr_df.to_string())

    # Evaluate Predictor Models M0 - M4
    print("\n==========================================================================")
    print("PREDICTOR MODEL COMPARISON (Held-Out Cross-Validation)")
    print("==========================================================================")

    predictors = {
        "M0 (Stage Only)": ["checkpoint_stage"],
        "M1 (Baseline Accuracy Only)": ["pre_pass_at_1"],
        "M2 (Policy Entropy Only)": ["policy_entropy"],
        "M3 (Gradient Alignment Only)": ["gradient_alignment"],
        "M4 (Full Diagnostic Model)": ["gradient_alignment", "policy_entropy", "pre_pass_at_1", "gradient_norm"],
    }

    for p_name, p_cols in predictors.items():
        X = df[p_cols].values
        y = df["rl_gain"].values
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        r2 = model.score(X, y)
        print(f"  {p_name:<35} | R^2 = {r2:+.4f}")

    # Cross-Model Transfer (SmolLM-135M -> distilgpt2)
    print("\n==========================================================================")
    print("CROSS-MODEL ZERO-SHOT TRANSFER (SmolLM-135M -> distilgpt2)")
    print("==========================================================================")

    df_smol = df[df["model_name"] == "HuggingFaceTB/SmolLM-135M"]
    df_distil = df[df["model_name"] == "distilgpt2"]

    feature_cols = ["gradient_alignment", "policy_entropy", "pre_pass_at_1", "gradient_norm"]

    if len(df_smol) > 0 and len(df_distil) > 0:
        X_train, y_train = df_smol[feature_cols].values, df_smol["rl_gain"].values
        X_test, y_test = df_distil[feature_cols].values, df_distil["rl_gain"].values

        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        transfer_r2 = 1.0 - (ss_res / max(1e-8, ss_tot))
        transfer_rho, _ = stats.spearmanr(y_test, y_pred)

        print(f"  Train on SmolLM-135M (Fit R^2: {clf.score(X_train, y_train):.4f})")
        print(f"  Zero-Shot Test on distilgpt2 | R^2 = {transfer_r2:+.4f} | Spearman Rho = {transfer_rho:+.4f}")
    else:
        print("  Insufficient model family splits for cross-model transfer test.")


if __name__ == "__main__":
    p_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/empirical/rl_plasticity_dataset.parquet"
    analyze_empirical_dataset(p_path)
