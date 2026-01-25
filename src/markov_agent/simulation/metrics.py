from collections import defaultdict

from markov_agent.simulation.runner import SimulationResult


def calculate_pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k (Probability of at least one success).
    n: total samples generated
    c: number of correct samples
    k: hypothetical budget (k)
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


def calculate_pass_pow_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass^k (Probability that all k samples are correct).
    n: total samples generated
    c: number of correct samples
    k: hypothetical budget (k)
    """
    if c < k:
        return 0.0

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

    success_combinations = comb(c, k)
    return success_combinations / total_combinations


def calculate_metrics(results: list[SimulationResult]):
    """
    Calculates:
    - Accuracy (Global pass rate, effectively pass@1 averaged)
    - Consistency (pass^k): Probability that all samples in a budget k are correct.
    - Reliability (pass@k): Probability that at least one sample in a budget k is
      correct.
    """
    if not results:
        return {
            "accuracy": 0.0,
            "consistency": 0.0,
            "reliability": 0.0,
            "total_cases": 0,
            "total_runs": 0,
        }

    # Group by case_id
    cases = defaultdict(list)
    for r in results:
        cases[r.case_id].append(r)

    total_cases = len(cases)
    total_runs = len(results)
    total_successes = sum(1 for r in results if r.success)

    consistent_cases = 0  # 100% success for all n_runs
    reliable_cases = 0  # at least one success in n_runs

    for _case_id, case_results in cases.items():
        case_runs = len(case_results)
        case_successes = sum(1 for r in case_results if r.success)

        if case_successes == case_runs:
            consistent_cases += 1

        if case_successes > 0:
            reliable_cases += 1

    accuracy = total_successes / total_runs if total_runs > 0 else 0.0
    # Global metrics across the whole simulation (where k=n_runs per case)
    global_consistency = consistent_cases / total_cases if total_cases > 0 else 0.0
    global_reliability = reliable_cases / total_cases if total_cases > 0 else 0.0

    # Calculate pass@k and pass^k estimates for various k
    pass_at_k_scores = {}
    pass_pow_k_scores = {}

    if cases:
        max_runs = max(len(results) for results in cases.values())
        k_values_to_test = [k for k in [1, 2, 5, 10, 20] if k <= max_runs]
    else:
        k_values_to_test = []

    for k in k_values_to_test:
        sum_at_k = 0.0
        sum_pow_k = 0.0
        valid_cases_for_k = 0

        for _case_id, case_results in cases.items():
            n = len(case_results)
            if n < k:
                continue

            valid_cases_for_k += 1
            c = sum(1 for r in case_results if r.success)
            sum_at_k += calculate_pass_at_k_estimator(n, c, k)
            sum_pow_k += calculate_pass_pow_k_estimator(n, c, k)

        pass_at_k_scores[f"pass@{k}"] = (
            sum_at_k / valid_cases_for_k if valid_cases_for_k > 0 else 0.0
        )
        pass_pow_k_scores[f"pass^{k}"] = (
            sum_pow_k / valid_cases_for_k if valid_cases_for_k > 0 else 0.0
        )

    return {
        "accuracy": accuracy,
        "consistency": global_consistency,
        "reliability": global_reliability,
        "pass_at_k": pass_at_k_scores,
        "pass_pow_k": pass_pow_k_scores,
        "total_cases": total_cases,
        "total_runs": total_runs,
        "consistent_cases": consistent_cases,
    }
