import logging

import pytest
from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import FunctionalNode


class MyState(BaseState):
    query: str = ""
    answer: str = ""
    score: int = 0
    notes: list[str] = Field(default_factory=list)


class MyOutput(BaseModel):
    answer: str
    score: int


@pytest.mark.asyncio
async def test_fluent_graph_building():
    graph = Graph(name="test_graph", state_type=MyState)

    @graph.task
    def start_node(state: MyState) -> dict:
        return {"query": "hello"}

    @graph.task
    def end_node(state: MyState) -> MyState:
        return state.update(answer="world")

    graph.add_transition("start_node", "end_node")

    assert "start_node" in graph.nodes
    assert "end_node" in graph.nodes
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "start_node"
    assert graph.edges[0].target == "end_node"

    state = MyState()
    final_state = await graph.run(state)
    assert final_state.query == "hello"
    assert final_state.answer == "world"


@pytest.mark.asyncio
async def test_graph_chain():
    graph = Graph(name="chain_graph", state_type=MyState)

    @graph.task
    def a(s: MyState) -> dict:
        return {"query": "a"}

    @graph.task
    def b(s: MyState) -> dict:
        return {"answer": "b"}

    @graph.task
    def c(s: MyState) -> dict:
        return {"score": 100}

    graph.chain(["a", "b", "c"])

    assert len(graph.edges) == 2

    final_state = await graph.run(MyState())
    assert final_state.query == "a"
    assert final_state.answer == "b"
    assert final_state.score == 100


@pytest.mark.asyncio
async def test_smart_state_mapping():
    node = ProbabilisticNode(
        name="test_ppu",
        adk_config=ADKConfig(model_name="mock"),
        prompt_template="test",
        output_schema=MyOutput,
        state_type=MyState,
    )

    state = MyState(query="what is 1+1?")
    output = MyOutput(answer="2", score=10)

    new_state = node.parse_result(state, output)

    assert new_state.answer == "2"
    assert new_state.score == 10
    assert new_state.query == "what is 1+1?"


@pytest.mark.asyncio
async def test_smart_state_mapping_no_match(caplog):
    class OtherOutput(BaseModel):
        something_else: str

    node = ProbabilisticNode(
        name="test_ppu",
        adk_config=ADKConfig(model_name="mock"),
        prompt_template="test",
        output_schema=OtherOutput,
        state_type=MyState,
    )

    state = MyState()
    output = OtherOutput(something_else="val")

    with caplog.at_level(logging.WARNING):
        new_state = node.parse_result(state, output)

    assert "No matching fields found" in caplog.text
    assert isinstance(new_state, MyState)


@pytest.mark.asyncio
async def test_functional_node_async():
    async def async_task(state: MyState) -> dict:
        import asyncio

        await asyncio.sleep(0.01)
        return {"answer": "async_result"}

    node = FunctionalNode(name="async_node", func=async_task, state_type=MyState)
    state = MyState()
    new_state = await node.execute(state)
    assert new_state.answer == "async_result"


def test_add_transition_with_condition():
    graph = Graph(name="cond_graph", state_type=MyState)
    graph.add_node(FunctionalNode(name="start", func=lambda s: s))
    graph.add_node(FunctionalNode(name="high", func=lambda s: s))
    graph.add_node(FunctionalNode(name="low", func=lambda s: s))

    graph.add_transition("start", "high", condition=lambda s: s.score > 50)
    graph.add_transition("start", "low", condition=lambda s: s.score <= 50)

    assert len(graph.edges) == 2
    assert graph.edges[0].condition is not None
