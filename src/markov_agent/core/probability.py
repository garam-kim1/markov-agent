import math
from typing import Any


class LogProb:
    """Helper for log-space probability arithmetic to prevent underflow."""

    @staticmethod
    def from_float(p: float) -> float:
        """Convert [0, 1] probability to log space (-inf, 0]."""
        if p <= 0:
            return -float("inf")
        return math.log(p)

    @staticmethod
    def to_float(log_p: float) -> float:
        """Convert log space back to [0, 1] probability."""
        if log_p == -float("inf"):
            return 0.0
        return math.exp(log_p)

    @staticmethod
    def multiply(log_p1: float, log_p2: float) -> float:
        """Multiply two probabilities in log space (addition)."""
        return log_p1 + log_p2

    @staticmethod
    def add(log_p1: float, log_p2: float) -> float:
        """Add two probabilities in log space (log-sum-exp).

        log(p1 + p2) = log(exp(log_p1) + exp(log_p2))
        """
        if log_p1 == -float("inf"):
            return log_p2
        if log_p2 == -float("inf"):
            return log_p1

        # log(exp(a) + exp(b)) = a + log(1 + exp(b - a)) where a > b
        if log_p1 < log_p2:
            log_p1, log_p2 = log_p2, log_p1

        return log_p1 + math.log1p(math.exp(log_p2 - log_p1))


def calculate_entropy(distribution: dict[Any, float]) -> float:
    """Calculate Shannon entropy of a distribution.

    Formula: H(P) = -sum(p_i * log2(p_i))
    """
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def normalize_distribution(dist: dict[Any, float]) -> dict[Any, float]:
    """Normalize a dictionary of raw weights to a probability distribution.

    If the total weight is 0, returns a uniform distribution.
    """
    total = sum(dist.values())
    if total <= 0:
        if not dist:
            return {}
        uniform_p = 1.0 / len(dist)
        return dict.fromkeys(dist, uniform_p)
    return {k: v / total for k, v in dist.items()}


def apply_temperature(dist: dict[Any, float], temperature: float) -> dict[Any, float]:
    """Apply temperature scaling to a distribution using Softmax-style scaling.

    Formula: p_i' = exp(log(p_i) / T) / sum(exp(log(p_j) / T))

    Args:
        dist: The original probability distribution.
        temperature: Scaling factor. T=1.0 is no change, T->0 is greedy (argmax),
                     T->inf is uniform.

    """
    if temperature <= 0:
        # Use greedy selection (argmax)
        max_val = max(dist.values())
        max_keys = [k for k, v in dist.items() if v == max_val]
        return normalize_distribution(
            {k: (1.0 if k in max_keys else 0.0) for k in dist}
        )

    if temperature == 1.0:
        return dist

    # Use log space for stability
    log_dist = {k: math.log(v) if v > 0 else -float("inf") for k, v in dist.items()}
    scaled_log_dist = {k: v / temperature for k, v in log_dist.items()}

    # Subtract max for numerical stability before exp (Softmax trick)
    max_log = max(scaled_log_dist.values())
    if max_log == -float("inf"):
        return normalize_distribution(dict.fromkeys(dist, 1.0))

    exp_dist = {k: math.exp(v - max_log) for k, v in scaled_log_dist.items()}
    return normalize_distribution(exp_dist)


def kl_divergence(
    p: dict[Any, float], q: dict[Any, float], epsilon: float = 1e-10
) -> float:
    """Calculate Kullback-Leibler divergence D_KL(P || Q).

    Measures how much policy P differs from reference policy Q.
    Formula: D_KL(P || Q) = sum(p_i * log2(p_i / q_i))
    """
    keys = set(p.keys()) | set(q.keys())
    div = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        # Use epsilon smoothing to avoid division by zero or log(0)
        pk_smooth = max(pk, epsilon)
        qk_smooth = max(qk, epsilon)
        if pk > 0:
            div += pk * math.log2(pk_smooth / qk_smooth)
    return div


def jensen_shannon_divergence(p: dict[Any, float], q: dict[Any, float]) -> float:
    """Calculate Jensen-Shannon divergence (symmetric bounded metric).

    Formula: JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    keys = set(p.keys()) | set(q.keys())
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    # No epsilon needed because M is guaranteed to be non-zero if P or Q are non-zero
    return 0.5 * kl_divergence(p, m, epsilon=0.0) + 0.5 * kl_divergence(
        q, m, epsilon=0.0
    )
