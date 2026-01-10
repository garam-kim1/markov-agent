from markov_agent.simulation.runner import SimulationResult


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates pass@k metric.
    n: total number of samples
    c: number of correct samples
    k: k in pass@k

    Formula: 1 - (comb(n-c, k) / comb(n, k))
    This is often approximated if n and c are large.
    For small n, c, we can use exact math.
    """

    if n - c < k:
        return 1.0

    def comb(n, k):
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        if k > n // 2:
            k = n - k

        numerator = 1
        for i in range(k):
            numerator = numerator * (n - i) // (i + 1)
        return numerator

    total_combinations = comb(n, k)
    if total_combinations == 0:
        return 0.0

    failed_combinations = comb(n - c, k)
    return 1.0 - (failed_combinations / total_combinations)


def calculate_metrics(results: list[SimulationResult]):
    """
    Calculates accuracy and consistency metrics from simulation results.
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    successes = sum(1 for r in results if r.success)
    accuracy = successes / total

    return {
        "accuracy": accuracy,
        "successes": successes,
        "total": total,
        "pass@1": calculate_pass_at_k(total, successes, 1),
        # Pass@k only makes sense if we have multiple runs for the same input
        # This is a simplified view
    }
