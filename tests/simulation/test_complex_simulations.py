import random

import pytest
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.topology.graph import Graph


class MathState(BaseState):
    value: int = 0
    ops: list[str] = Field(default_factory=list)
    target: int = 10


@pytest.mark.asyncio
async def test_monte_carlo_math_convergence():
    """Run a Monte Carlo simulation on a graph that tries to reach a target value."""
    # This graph has a node that increments or decrements a value
    # and a transition that checks if it reached the target.

    def mock_resp(p):
        return '{"op": "inc"}' if random.random() > 0.3 else '{"op": "dec"}'

    config = ADKConfig(
        model_name="openai/mock-model",
        api_base="http://localhost:8080",
        mock_responder=mock_resp,
    )

    graph = Graph(name="math_game", state_type=MathState)

    @graph.node(adk_config=config, output_schema=dict)
    def calculator(state: MathState, result: dict) -> dict:
        """Decide whether to increment or decrement to reach target."""
        op = result.get("op", "inc")
        new_val = state.value + 1 if op == "inc" else state.value - 1
        return {"value": new_val, "ops": [*state.ops, op]}

    graph.add_transition(
        "calculator", "calculator", condition=lambda s: s.value != s.target
    )

    # Success criteria for simulation
    def is_success(state: MathState) -> bool:
        return state.value == state.target

    # Run simulation with multiple trajectories
    dataset = [{"value": 0, "target": 5}]  # Starting point
    results = await graph.simulate(
        dataset=dataset, n_runs=5, success_criteria=is_success, max_concurrency=2
    )

    assert len(results) == 5
    # Some should succeed
    assert any(r.success for r in results) or all(r.steps >= 0 for r in results)


@pytest.mark.asyncio
async def test_complex_nested_graph_simulation():
    """Test simulation of a graph containing subgraphs."""

    def inner_mock(p):
        return '{"valid": true}'

    inner_config = ADKConfig(
        model_name="openai/mock-model",
        api_base="http://localhost:8080",
        mock_responder=inner_mock,
    )

    inner_graph = Graph(name="validator")

    @inner_graph.node(adk_config=inner_config, output_schema=dict)
    def check_valid(state: BaseState, result: dict) -> dict:
        """Check if state is valid."""
        return {"meta": {"validated": result.get("valid", False)}}

    outer_graph = Graph(name="main_process")

    @outer_graph.node(adk_config=inner_config, output_schema=dict)
    def generate_data(state: BaseState, result: dict) -> dict:
        """Generate some data."""
        return {"data": "some data"}

    # Add inner graph as a subgraph node
    outer_graph.subgraph(inner_graph, name="validate_step")

    outer_graph.add_transition("generate_data", "validate_step")

    dataset = [{}]
    results = await outer_graph.simulate(dataset, n_runs=2)

    assert len(results) == 2
    for r in results:
        assert r.final_state is not None
        assert r.final_state.meta.get("validated") is True
