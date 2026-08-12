"""
Orchestrates multi-seed baseline experiments (B0–B7), ablations, and generates analysis summaries.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.run_experiment import run_experiment
from adaptive_rl_forge.analysis.statistics import (
    bootstrap_confidence_interval,
    compute_correlations,
    fit_plasticity_regression,
)


def run_full_experimental_suite():
    baselines = [
        "B0_NTP",
        "B1_SFT",
        "B2_Sequential",
        "B3_EarlyRL",
        "B4_PeriodicRL",
        "B5_RandomRL",
        "B6_FixedMixture",
        "B7_CARLS",
    ]
    seeds = [42, 43, 44]
    output_dir = "artifacts/runs"

    results_list = []

    print("=== STARTING MULTI-SEED EXPERIMENTAL SUITE ===")
    for b in baselines:
        for seed in seeds:
            res = run_experiment(baseline=b, total_steps=120, seed=seed, output_dir=output_dir, model_scale="default")
            res["baseline"] = b
            res["seed"] = seed
            res["model_scale"] = "default"
            results_list.append(res)

    # Cross-Model Scale Runs for CARLS and Sequential
    for b in ["B2_Sequential", "B7_CARLS"]:
        for seed in [42, 43]:
            res = run_experiment(baseline=b, total_steps=120, seed=seed, output_dir=output_dir, model_scale="tiny")
            res["baseline"] = b
            res["seed"] = seed
            res["model_scale"] = "tiny"
            results_list.append(res)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv("artifacts/master_experiment_results.csv", index=False)
    print("Saved master experiment results to artifacts/master_experiment_results.csv")

    # Aggregate Statistics per Baseline
    summary_rows = []
    for b in baselines:
        sub = df_results[(df_results["baseline"] == b) & (df_results["model_scale"] == "default")]
        p1_vals = sub["pass_at_1"].values
        p4_vals = sub["pass_at_4"].values
        div_vals = sub["solution_diversity_ratio"].values
        flops_vals = sub["total_flops"].values
        toks_vals = sub["total_tokens"].values

        p1_mean, p1_low, p1_up = bootstrap_confidence_interval(p1_vals)
        p4_mean, p4_low, p4_up = bootstrap_confidence_interval(p4_vals)
        div_mean, _, _ = bootstrap_confidence_interval(div_vals)

        summary_rows.append({
            "Baseline": b,
            "Pass@1_Mean": p1_mean,
            "Pass@1_95CI": f"[{p1_low:.3f}, {p1_up:.3f}]",
            "Pass@4_Mean": p4_mean,
            "Pass@4_95CI": f"[{p4_low:.3f}, {p4_up:.3f}]",
            "Diversity_Ratio": div_mean,
            "Total_Tokens": int(np.mean(toks_vals)),
            "Total_FLOPs": float(np.mean(flops_vals)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("artifacts/baseline_summary_table.csv", index=False)
    print("\n=== BASELINE PERFORMANCE SUMMARY ===")
    print(summary_df.to_string())

    # Generate Figures
    os.makedirs("artifacts/figures", exist_ok=True)

    # Figure 1: Baseline Comparison Bar Plot
    plt.figure(figsize=(10, 5))
    x_coords = np.arange(len(baselines))
    plt.bar(x_coords - 0.2, summary_df["Pass@1_Mean"], width=0.4, label="Pass@1", color="#1f77b4")
    plt.bar(x_coords + 0.2, summary_df["Pass@4_Mean"], width=0.4, label="Pass@4", color="#ff7f0e")
    plt.xticks(x_coords, baselines, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Compute-Normalized Benchmark Accuracy Across Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/figures/fig1_baseline_accuracy.png", dpi=300)
    plt.close()

    # Figure 2: Compute Pareto Curve (Tokens vs Pass@4)
    plt.figure(figsize=(8, 5))
    for b in baselines:
        sub = summary_df[summary_df["Baseline"] == b]
        plt.scatter(sub["Total_Tokens"], sub["Pass@4_Mean"], label=b, s=100)
    plt.xlabel("Total Training Tokens")
    plt.ylabel("Pass@4 Accuracy")
    plt.title("Compute Efficiency Pareto Frontier")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("artifacts/figures/fig2_compute_pareto.png", dpi=300)
    plt.close()

    print("Generated plots in artifacts/figures/")


if __name__ == "__main__":
    run_full_experimental_suite()
