import pytest
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode
from markov_agent.topology.graph import Graph
from markov_agent.simulation.metrics import calculate_metrics

class SimState(BaseState):
    val: int = 0

class ErrorNode(BaseNode[SimState]):
    async def execute(self, state: SimState) -> SimState:
        if state.val == -1:
            raise ValueError("Simulated Crash")
        return state.update(val=state.val + 1)

@pytest.mark.asyncio
async def test_monte_carlo_errors():
    node = ErrorNode("err")
    graph = Graph(name="sim_err", nodes={"err": node}, edges=[], entry_point="err")
    
    # Dataset: one safe case, one error case
    dataset = [SimState(val=0), SimState(val=-1)]
    
    runner = MonteCarloRunner(
        graph=graph, 
        dataset=dataset, 
        n_runs=1,
        success_criteria=lambda s: True
    )
    
    results = await runner.run_simulation()
    
    assert len(results) == 2
    
    # Case 0: Success
    assert results[0].success is True
    assert results[0].error is None
    
    # Case 1: Exception
    assert results[1].success is False
    assert results[1].error == "Simulated Crash"
    assert results[1].final_state is None

@pytest.mark.asyncio
async def test_monte_carlo_metrics_failure():
    node = ErrorNode("inc")
    graph = Graph(name="sim_fail", nodes={"inc": node}, edges=[], entry_point="inc")
    
    # Dataset: 2 items
    dataset = [SimState(val=0), SimState(val=0)]
    
    # Success only if val > 10 (which is impossible here, val becomes 1)
    runner = MonteCarloRunner(
        graph=graph,
        dataset=dataset,
        n_runs=1,
        success_criteria=lambda s: s.val > 10
    )
    
    results = await runner.run_simulation()
    metrics = calculate_metrics(results)
    
    assert metrics["accuracy"] == 0.0
    assert metrics["total_cases"] == 2
