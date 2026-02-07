from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner, SimulationResult


class ConsistencyGatekeeper(BaseModel):
    """Formal Gatekeeper for Production Deployment.

    Validates that a system meets the pass ∧ k (Consistency) threshold.
    """

    batch_size: int = Field(default=10, description="The 'k' in pass ∧ k.")
    required_stability: float = Field(
        default=1.0, description="Required success rate (usually 1.0 for enterprise)."
    )

    async def validate_topology(
        self,
        runner: MonteCarloRunner,
        test_cases: list[dict[str, Any]],
        evaluator: Callable[[Any], bool],
    ) -> bool:
        """Run batch stress tests and return True only if stability requirements are met."""
        results: list[SimulationResult] = []

        # Run each test case 'batch_size' times
        for case in test_cases:
            case_results = await runner.run_simulation(
                n_runs=self.batch_size, test_cases=[case], evaluator=evaluator
            )
            results.extend(case_results)

        metrics = calculate_metrics(results)
        # Check global consistency (pass^k)
        return metrics["consistency"] >= self.required_stability
