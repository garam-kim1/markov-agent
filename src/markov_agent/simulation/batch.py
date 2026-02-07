from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner


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
        test_cases: list[Any],
        evaluator: Callable[[Any], bool],
    ) -> bool:
        """Run batch stress tests and return True only if stability requirements are met."""
        # Update runner configuration
        runner.n_runs = self.batch_size
        runner.dataset = test_cases
        runner.success_criteria = evaluator

        results = await runner.run_simulation()

        metrics = calculate_metrics(results)
        # Check global consistency (pass^k)
        return metrics["consistency"] >= self.required_stability
