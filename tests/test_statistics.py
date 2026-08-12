import numpy as np
import pandas as pd
import pytest
from adaptive_rl_forge.analysis.statistics import (
    compute_correlations,
    bootstrap_confidence_interval,
    fit_plasticity_regression,
)


def test_statistics_and_bootstrapping():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_val, lower, upper = bootstrap_confidence_interval(data, num_bootstraps=100, ci=95.0)
    assert lower <= mean_val <= upper

    df = pd.DataFrame({
        "grad_align": [0.1, 0.4, 0.6, 0.8],
        "entropy": [2.5, 2.1, 1.8, 1.2],
        "rl_gain": [0.05, 0.20, 0.35, 0.50],
    })

    corr_df = compute_correlations(df, ["grad_align", "entropy"], "rl_gain")
    assert len(corr_df) == 2
    assert "pearson_r" in corr_df.columns

    reg_res = fit_plasticity_regression(df, ["grad_align", "entropy"], "rl_gain")
    assert "r2_score" in reg_res
    assert "coefficients" in reg_res
