"""
CARLS (Capability-Aware Reinforcement Learning Scheduler) Controller module.
Outputs dynamic objective mixture weights: (alpha_NTP, beta_SFT, gamma_RL) subject to sum = 1.0.
"""

from typing import Dict, Any, Tuple


class CARLSv0Controller:
    """
    CARLS-v0: Rule-based dynamic scheduling controller.
    Uses gradient alignment and entropy thresholds to trigger RL excursions.
    """

    def __init__(
        self,
        min_grad_align: float = 0.05,
        min_entropy: float = 1.0,
        max_pass_k: float = 0.8,
    ):
        self.min_grad_align = min_grad_align
        self.min_entropy = min_entropy
        self.max_pass_k = max_pass_k

    def compute_allocation(self, signals: Dict[str, float]) -> Tuple[float, float, float]:
        grad_align = signals.get("gradient_alignment", 0.0)
        entropy = signals.get("policy_entropy", 2.0)
        pass_k = signals.get("pass_at_k", 0.0)

        # High gradient alignment & adequate policy entropy -> High RL allocation (excursion)
        if grad_align >= self.min_grad_align and entropy >= self.min_entropy and pass_k < self.max_pass_k:
            # Active RL excursion phase
            alpha_ntp = 0.2
            beta_sft = 0.1
            gamma_rl = 0.7
        elif pass_k >= self.max_pass_k:
            # High capability baseline -> focus on NTP stability
            alpha_ntp = 0.8
            beta_sft = 0.1
            gamma_rl = 0.1
        else:
            # Low gradient alignment / low entropy -> SFT bridge + NTP pre-training
            alpha_ntp = 0.6
            beta_sft = 0.3
            gamma_rl = 0.1

        return alpha_ntp, beta_sft, gamma_rl


class CARLSv1Controller:
    """
    CARLS-v1: Learned plastic gain predictor controller.
    Uses linear combination of signals to estimate expected RL gain and dynamically assign weights.
    """

    def __init__(self):
        # Initial weights learned from checkpoint plasticity study regression
        self.w_grad = 0.45
        self.w_entropy = 0.35
        self.w_passk = -0.20
        self.bias = 0.10

    def predict_rl_gain(self, signals: Dict[str, float]) -> float:
        grad_align = signals.get("gradient_alignment", 0.0)
        entropy = signals.get("policy_entropy", 0.0)
        pass_k = signals.get("pass_at_k", 0.0)

        score = (
            self.w_grad * grad_align
            + self.w_entropy * (entropy / 3.0)
            + self.w_passk * pass_k
            + self.bias
        )
        return float(score)

    def compute_allocation(self, signals: Dict[str, float]) -> Tuple[float, float, float]:
        predicted_gain = self.predict_rl_gain(signals)

        if predicted_gain > 0.3:
            gamma_rl = min(0.8, max(0.4, predicted_gain))
            alpha_ntp = (1.0 - gamma_rl) * 0.7
            beta_sft = 1.0 - alpha_ntp - gamma_rl
        else:
            gamma_rl = 0.1
            beta_sft = 0.2
            alpha_ntp = 0.7

        return alpha_ntp, beta_sft, gamma_rl
