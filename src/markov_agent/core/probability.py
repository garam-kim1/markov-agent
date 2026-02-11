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
    """Calculate Shannon entropy of a distribution."""
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)
