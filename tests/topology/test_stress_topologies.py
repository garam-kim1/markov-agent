import pytest
from pydantic import Field

from markov_agent.containers.parallel import ParallelNode
from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import FunctionalNode


class StressState(BaseState):
    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append"}
    )
    counter: int = 0
    total_sum: int = 0


@pytest.mark.asyncio
async def test_parallel_append_merge():
    """Test that multiple parallel branches appending to the same list are all captured."""
    graph = Graph(name="parallel_append", state_type=StressState)

    def branch_1(s: StressState):
        return {"logs": ["B1"]}

    def branch_2(s: StressState):
        return {"logs": ["B2"]}

    def branch_3(s: StressState):
        return {"logs": ["B3"]}

    graph.parallel([branch_1, branch_2, branch_3], name="p_append")

    state = StressState()
    result = await graph.run(state)

    assert len(result.logs) == 3
    assert "B1" in result.logs
    assert "B2" in result.logs
    assert "B3" in result.logs


@pytest.mark.asyncio
async def test_nested_loop_parallel():
    """Test a loop that contains a parallel block."""
    graph = Graph(name="nested_stress", state_type=StressState)

    # Inner parallel block: increment counter and add log
    def p1(s: StressState):
        return {"counter": s.counter + 1, "logs": [f"P1_{s.counter}"]}

    def p2(s: StressState):
        return {"counter": s.counter + 1, "logs": [f"P2_{s.counter}"]}

    p_block = ParallelNode(
        name="inner_p",
        nodes=[
            FunctionalNode(name="p1", func=p1, state_type=StressState),
            FunctionalNode(name="p2", func=p2, state_type=StressState),
        ],
        state_type=StressState,
    )

    # Outer loop: run 2 times
    graph.loop(body=p_block, condition=lambda s: s.counter >= 4, name="outer_loop")

    state = StressState()
    result = await graph.run(state)

    assert result.counter >= 4
    # Each iteration adds 2 logs. 4 iterations = 8 logs.
    assert len(result.logs) == 8


@pytest.mark.asyncio
async def test_loop_with_if_else():
    """Test a loop that contains an if_else branch."""
    # We can't easily put an if_else INSIDE a LoopNode body because LoopNode takes a single Node.
    # But we can use a NestedGraphNode as the body.

    body_graph = Graph(name="body", state_type=StressState)
    body_graph.task(lambda s: {"counter": s.counter + 1}, name="inc")
    body_graph.task(lambda s: {"logs": ["High"]}, name="high")
    body_graph.task(lambda s: {"logs": ["Low"]}, name="low")

    body_graph.if_else(
        condition=lambda s: s.counter > 2,
        then_node="high",
        else_node="low",
        source="inc",
    )

    main = Graph(name="main", state_type=StressState)
    main.loop(body=body_graph, condition=lambda s: s.counter >= 4, name="main_loop")

    state = StressState()
    result = await main.run(state)

    assert result.counter == 4
    # Iteration 1: counter=1, logs=["Low"]
    # Iteration 2: counter=2, logs=["Low"]
    # Iteration 3: counter=3, logs=["High"]
    # Iteration 4: counter=4, logs=["High"]
    assert result.logs == ["Low", "Low", "High", "High"]
