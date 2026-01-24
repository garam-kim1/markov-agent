import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph

StateT = TypeVar("StateT", bound=BaseState)


class SimulationResult(BaseModel):
    case_id: str = ""
    input_state: Any = None
    final_state: Any
    success: bool
    error: str | None = None
    steps: int = 0


class MonteCarloRunner(BaseModel):
    """
    Runs the dataset through the graph N times to ensure reliability.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Graph
    dataset: list[BaseState]
    n_runs: int = 1  # Number of times to run EACH item in dataset
    success_criteria: Callable[[BaseState], bool] = lambda s: True

    async def run_simulation(self) -> list[SimulationResult]:
        all_results = []

        tasks = []
        for i, state in enumerate(self.dataset):
            case_id = f"case_{i}"
            for _ in range(self.n_runs):
                # We must deep copy or re-instantiate the state for each run
                # Pydantic model_copy(deep=True) is useful here
                tasks.append(self._run_single(state.model_copy(deep=True), case_id))

        all_results = await asyncio.gather(*tasks)
        return all_results

    async def _run_single(
        self, initial_state: StateT, case_id: str
    ) -> SimulationResult:
        try:
            final_state = await self.graph.run(initial_state)
            success = self.success_criteria(final_state)
            return SimulationResult(
                case_id=case_id,
                input_state=initial_state,
                final_state=final_state,
                success=success,
            )
        except Exception as e:
            return SimulationResult(
                case_id=case_id,
                input_state=initial_state,
                final_state=None,
                success=False,
                error=str(e),
            )
