import pytest

from markov_agent.core.state import BaseState
from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class SimState(BaseState):
    value: int = 0


class IncrementNode(BaseNode[SimState]):
    async def execute(self, state: SimState) -> SimState:
        return state.update(value=state.value + 1)


@pytest.mark.asyncio
async def test_monte_carlo_runner():
    node = IncrementNode(name="inc")
    graph = Graph(name="test_sim", nodes={"inc": node}, edges=[], entry_point="inc")

    dataset = [SimState(value=i) for i in range(5)]
    runner = MonteCarloRunner(
        graph=graph,
        dataset=dataset,
        n_runs=2,
        success_criteria=lambda s: s.value > 0,
    )

    results = await runner.run_simulation()
    assert len(results) == 10  # 5 items * 2 runs

    metrics = calculate_metrics(results)
    assert metrics["total_runs"] == 10
    # Success if value > 0.
    # Inputs: 0, 1, 2, 3, 4.
    # After increment: 1, 2, 3, 4, 5. All > 0.
    assert metrics["accuracy"] == 1.0
    assert metrics["consistency"] == 1.0
    assert metrics["total_cases"] == 5
