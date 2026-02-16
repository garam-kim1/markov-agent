from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from markov_agent.core.state import BaseState

if TYPE_CHECKING:
    from collections.abc import Callable

    from markov_agent.simulation.runner import SimulationResult
    from markov_agent.topology.graph import Graph

StateT = TypeVar("StateT", bound=BaseState)


@dataclass
class Scenario:
    """Defines a specific 'What-If' scenario for simulation."""

    name: str
    description: str = ""
    state_overrides: dict[str, Any] = field(default_factory=dict)
    n_runs: int = 5

    # Optional: Override graph parameters (e.g., strict_markov)
    graph_config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregated results for a specific scenario."""

    scenario_name: str
    success_rate: float
    avg_steps: float
    avg_reward: float
    runs: list[SimulationResult]

    def __str__(self) -> str:
        """Return a string representation of the result."""
        return (
            f"[{self.scenario_name}] Success: {self.success_rate:.1%} | "
            f"Reward: {self.avg_reward:.2f} | Steps: {self.avg_steps:.1f}"
        )


class ScenarioManager:
    """Manages the execution and comparison of multiple scenarios."""

    def __init__(self, graph: Graph):
        self.graph = graph

    async def run_scenarios(
        self,
        base_state: StateT,
        scenarios: list[Scenario],
        success_criteria: Callable[[Any], bool] | None = None,
        max_concurrency: int = 10,
    ) -> list[ScenarioResult]:
        """Run multiple scenarios in sequence and aggregate results."""
        results = []

        for scenario in scenarios:
            # 1. Prepare State
            # Create a clean copy of the base state
            current_state = base_state.update(**scenario.state_overrides)

            # 2. Apply Graph Overrides (Temporary)
            original_config = {
                k: getattr(self.graph, k) for k in scenario.graph_config_overrides
            }
            for k, v in scenario.graph_config_overrides.items():
                setattr(self.graph, k, v)

            try:
                # 3. Run Simulation
                sim_results = await self.graph.simulate(
                    dataset=[current_state],  # Simulate expects a list of inputs
                    n_runs=scenario.n_runs,
                    success_criteria=success_criteria,
                    max_concurrency=max_concurrency,
                )

                # 4. Aggregate
                n_success = sum(1 for r in sim_results if r.success)
                total_steps = sum(len(r.trajectory) for r in sim_results)

                total_reward = 0.0
                for r in sim_results:
                    # Check explicit reward field first, then meta
                    if hasattr(r.final_state, "reward"):
                        total_reward += r.final_state.reward
                    else:
                        total_reward += r.final_state.meta.get("reward", 0.0)

                results.append(
                    ScenarioResult(
                        scenario_name=scenario.name,
                        success_rate=n_success / len(sim_results)
                        if sim_results
                        else 0.0,
                        avg_steps=total_steps / len(sim_results)
                        if sim_results
                        else 0.0,
                        avg_reward=total_reward / len(sim_results)
                        if sim_results
                        else 0.0,
                        runs=sim_results,
                    )
                )

            finally:
                # Restore Graph Config
                for k, v in original_config.items():
                    setattr(self.graph, k, v)

        return results
