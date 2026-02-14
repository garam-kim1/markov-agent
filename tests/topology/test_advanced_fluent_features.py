import pytest
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import FunctionalNode


class FluentState(BaseState):
    counter: int = 0
    is_valid: bool = False
    feedback: str = ""
    path: list[str] = Field(default_factory=list)


@pytest.mark.asyncio
async def test_fluent_loop():

    graph = Graph(name="loop_graph", state_type=FluentState)

    # Body increases counter

    def body_func(s: FluentState):

        return {"counter": s.counter + 1, "path": [*s.path, "loop"]}

    # Stop when counter == 3

    graph.loop(body=body_func, condition=lambda s: s.counter >= 3, name="my_loop")

    # Start -> Loop

    graph.add_node(FunctionalNode(name="start", func=lambda s: s))

    graph.add_transition("start", "my_loop")

    state = FluentState()

    result = await graph.run(state)

    assert result.counter == 3

    assert result.path == ["loop", "loop", "loop"]


@pytest.mark.asyncio
async def test_fluent_self_correction():
    graph = Graph(name="sc_graph", state_type=FluentState)

    # Primary attempt
    def primary(s: FluentState):
        # Fail first time, succeed second time
        valid = s.counter > 0
        return {
            "counter": s.counter + 1,
            "is_valid": valid,
            "path": [*s.path, "primary"],
        }

    # Critique
    def critique(s: FluentState):
        return {
            "is_valid": s.is_valid,
            "feedback": "Need more counter" if not s.is_valid else "OK",
        }

    sc_node = graph.self_correction(
        primary=FunctionalNode(name="p", func=primary),
        critique=FunctionalNode(name="c", func=critique),
        max_retries=2,
    )

    graph.add_node(FunctionalNode(name="start", func=lambda s: s))
    graph.add_transition("start", sc_node.name)

    state = FluentState()
    result = await graph.run(state)

    # Loop 1: Primary (counter=1, valid=True), Critique (valid=True) -> STOP
    # Wait, counter starts at 0.
    # Attempt 0: Primary (counter=1, is_valid=False), Critique (is_valid=False)
    # Attempt 1: Primary (counter=2, is_valid=True), Critique (is_valid=True) -> STOP
    assert result.counter == 2
    assert result.is_valid is True
    assert "primary" in result.path
    assert len([p for p in result.path if p == "primary"]) == 2


@pytest.mark.asyncio
async def test_fluent_if_else():
    graph = Graph(name="if_else_graph", state_type=FluentState)

    graph.task(lambda s: {"counter": 100}, name="start")
    graph.task(lambda s: {"path": ["high"]}, name="high_node")
    graph.task(lambda s: {"path": ["low"]}, name="low_node")

    # If counter > 50 go to high_node, else low_node
    graph.if_else(
        condition=lambda s: s.counter > 50,
        then_node="high_node",
        else_node="low_node",
        source="start",
    )

    state = FluentState()
    result = await graph.run(state)
    assert result.path == ["high"]

    # Test else branch
    graph_low = Graph(name="if_else_low", state_type=FluentState)
    graph_low.task(lambda s: {"counter": 10}, name="start")
    graph_low.task(lambda s: {"path": ["high"]}, name="high_node")
    graph_low.task(lambda s: {"path": ["low"]}, name="low_node")
    graph_low.if_else(
        condition=lambda s: s.counter > 50,
        then_node="high_node",
        else_node="low_node",
        source="start",
    )

    result_low = await graph_low.run(FluentState())
    assert result_low.path == ["low"]
