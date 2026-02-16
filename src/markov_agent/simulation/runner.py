import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from markov_agent.core.monitoring import memory_guard
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
    trajectory: list[Any] = Field(default_factory=list)


class MonteCarloRunner[StateT: BaseState](BaseModel):
    """Runs the dataset through the graph N times to ensure reliability."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Graph
    dataset: list[StateT]
    n_runs: int = 1  # Number of times to run EACH item in dataset
    success_criteria: Callable[[StateT], bool] = Field(default=lambda _: True)
    max_concurrency: int = 10

    async def run_simulation(self) -> list[SimulationResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _sem_run(state: StateT, case_id: str) -> SimulationResult:
            async with semaphore:
                # Check memory before starting
                await memory_guard()
                return await self._run_single(state, case_id)

        tasks = [
            _sem_run(state.model_copy(deep=True), f"case_{i}")
            for i, state in enumerate(self.dataset)
            for _ in range(self.n_runs)
        ]

        return await asyncio.gather(*tasks)

    async def _run_single(
        self,
        initial_state: StateT,
        case_id: str,
    ) -> SimulationResult:
        try:
            final_state = await self.graph.run(initial_state)
            success = self.success_criteria(final_state)
            trajectory = getattr(final_state, "history", [])
            return SimulationResult(
                case_id=case_id,
                input_state=initial_state,
                final_state=final_state,
                success=success,
                trajectory=trajectory,
                steps=len(trajectory),
            )
        except Exception as e:
            return SimulationResult(
                case_id=case_id,
                input_state=initial_state,
                final_state=None,
                success=False,
                error=str(e),
            )
