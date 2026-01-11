from collections import defaultdict

from markov_agent.simulation.runner import SimulationResult


def calculate_pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.
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


def calculate_metrics(results: list[SimulationResult]):
    """
    Calculates:
    - Accuracy (Global pass rate, effectively pass@1 averaged)
    - Consistency (pass^k): % of cases that succeeded in ALL runs.
    - Reliability (pass@k): % of cases that succeeded in AT LEAST ONE run.
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

    consistent_cases = 0
    reliable_cases = 0

    for _case_id, case_results in cases.items():
        case_runs = len(case_results)
        case_successes = sum(1 for r in case_results if r.success)

        if case_successes == case_runs:
            consistent_cases += 1

        if case_successes > 0:
            reliable_cases += 1

    accuracy = total_successes / total_runs if total_runs > 0 else 0.0
    consistency = consistent_cases / total_cases if total_cases > 0 else 0.0
    reliability = reliable_cases / total_cases if total_cases > 0 else 0.0

    return {
        "accuracy": accuracy,  # pass@1
        "consistency": consistency,  # pass^k (Strict Consistency)
        "reliability": reliability,  # At least one success
        "total_cases": total_cases,
        "total_runs": total_runs,
        "consistent_cases": consistent_cases,
    }
