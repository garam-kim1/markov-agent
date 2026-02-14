import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import FunctionalNode


class SimpleState(BaseState):
    counter: int = 0
    flag: bool = False


@pytest.mark.asyncio
async def test_graph_connect_flow():
    g = Graph("TestGraph", state_type=SimpleState)

    n1 = FunctionalNode("n1", lambda s: s.update(counter=s.counter + 1))
    n2 = FunctionalNode("n2", lambda s: s.update(counter=s.counter + 1))
    n3 = FunctionalNode("n3", lambda s: s.update(counter=s.counter + 1))

    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    # Test connect with flow
    g.connect(n1 >> n2 >> n3)

    assert len(g.edges) == 2
    assert g.edges[0].source == "n1"
    assert g.edges[0].target == "n2"
    assert g.edges[1].source == "n2"
    assert g.edges[1].target == "n3"


@pytest.mark.asyncio
async def test_graph_route_method():
    g = Graph("RouteGraph", state_type=SimpleState)

    @g.task
    def start(s: SimpleState):
        return s

    @g.task
    def branch_a(s: SimpleState):
        return s.update(flag=True)

    @g.task
    def branch_b(s: SimpleState):
        return s.update(flag=False)

    g.route(
        "start",
        {
            "branch_a": lambda s: s.counter > 0,
            "branch_b": None,  # Default
        },
    )

    assert len(g.edges) == 2

    # Test execution
    s1 = await g.run(SimpleState(counter=5))
    assert s1.flag is True

    s2 = await g.run(SimpleState(counter=-1))
    assert s2.flag is False


@pytest.mark.asyncio
async def test_markovian_aggregation():
    g = Graph("MarkovGraph", state_type=SimpleState)

    @g.task
    def start(s: SimpleState):
        return s

    @g.task
    def node_a(s: SimpleState):
        return s

    @g.task
    def node_b(s: SimpleState):
        return s

    # Add multiple probabilistic edges for the same source
    g.add_transition("start", lambda s: {"node_a": 0.5})
    g.add_transition("start", lambda s: {"node_b": 0.5})

    reached_a = False
    reached_b = False

    for _ in range(20):
        res = await g.run(SimpleState())
        last_node = res.history[-1]["node"]
        if last_node == "node_a":
            reached_a = True
        if last_node == "node_b":
            reached_b = True
        if reached_a and reached_b:
            break

    assert reached_a
    assert reached_b


@pytest.mark.asyncio
async def test_ppu_from_prompt():
    node = ProbabilisticNode.from_prompt("test_ppu", "Translate {{ text }} to French")
    assert node.name == "test_ppu"
    assert "Translate" in node.prompt_template


@pytest.mark.asyncio
async def test_graph_call_shortcut():
    g = Graph("CallGraph", state_type=SimpleState)

    @g.task
    def inc(s: SimpleState):
        return s.update(counter=s.counter + 1)

    state = await g(SimpleState(counter=10))
    assert state.counter == 11
