"""
Statistical analysis module for correlation analysis, regression modeling, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from typing import Dict, Any, Tuple, List


def compute_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Computes Pearson and Spearman correlations between checkpoint features and target RL gain.
    """
    results = []
    for col in feature_cols:
        x = df[col].values
        y = df[target_col].values

        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            p_corr, p_val = 0.0, 1.0
            s_corr, s_val = 0.0, 1.0
        else:
            p_corr, p_val = stats.pearsonr(x, y)
            s_corr, s_val = stats.spearmanr(x, y)

        results.append({
            "feature": col,
            "pearson_r": float(p_corr),
            "pearson_pvalue": float(p_val),
            "spearman_rho": float(s_corr),
            "spearman_pvalue": float(s_val),
        })

    return pd.DataFrame(results)


def bootstrap_confidence_interval(
    data: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes bootstrap mean and percentile confidence intervals (mean, ci_lower, ci_upper).
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(num_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    mean_val = float(np.mean(data))
    alpha = (100.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 - alpha))

    return mean_val, ci_lower, ci_upper


def fit_plasticity_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, Any]:
    """
    Fits a regularized linear model predicting subsequent RL gain from checkpoint signals.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    if len(df) < 4:
        return {"r2_score": 0.0, "coefficients": {col: 0.0 for col in feature_cols}}

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    r2 = model.score(X, y)

    coef_dict = {col: float(coef) for col, coef in zip(feature_cols, model.coef_)}
    coef_dict["intercept"] = float(model.intercept_)

    return {
        "r2_score": float(r2),
        "coefficients": coef_dict,
    }
