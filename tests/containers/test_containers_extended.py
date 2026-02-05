import pytest

from markov_agent.containers.loop import LoopNode
from markov_agent.containers.nested import NestedGraphNode
from markov_agent.containers.sequential import SequentialNode
from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class ExtendedState(BaseState):
    counter: int = 0
    trace: list[str] = []


class TraceNode(BaseNode[ExtendedState]):
    def __init__(self, name: str, increment: int = 0):
        super().__init__(name)
        self.increment = increment

    async def execute(self, state: ExtendedState) -> ExtendedState:
        return state.update(
            counter=state.counter + self.increment,
            trace=[*state.trace, self.name],
        )


@pytest.mark.asyncio
async def test_sequential_node():
    node_a = TraceNode("A", increment=1)
    node_b = TraceNode("B", increment=2)

    seq = SequentialNode(name="Seq", nodes=[node_a, node_b])

    initial = ExtendedState()
    final = await seq.execute(initial)

    assert final.counter == 3
    assert final.trace == ["A", "B"]


@pytest.mark.asyncio
async def test_loop_node_condition():
    """Loop until counter >= 3."""
    node_inc = TraceNode("Inc", increment=1)

    # Condition returns True to STOP
    def stop_condition(s: ExtendedState) -> bool:
        return s.counter >= 3

    loop = LoopNode(
        name="Loop",
        body=node_inc,
        condition=stop_condition,
        max_iterations=10,
    )

    initial = ExtendedState(counter=0)
    final = await loop.execute(initial)

    # Iteration 1: 0 -> 1 (Inc)
    # Iteration 2: 1 -> 2 (Inc)
    # Iteration 3: 2 -> 3 (Inc)
    # Iteration 4: Check 3 >= 3 -> True -> Break
    assert final.counter == 3
    assert final.trace == ["Inc", "Inc", "Inc"]


@pytest.mark.asyncio
async def test_loop_node_max_iterations():
    """Loop that never satisfies condition but hits max iterations."""
    node_inc = TraceNode("Inc", increment=1)

    # Condition never satisfied
    def never_stop(s: ExtendedState) -> bool:
        return False

    loop = LoopNode(
        name="LoopMax",
        body=node_inc,
        condition=never_stop,
        max_iterations=5,
    )

    initial = ExtendedState(counter=0)
    final = await loop.execute(initial)

    assert final.counter == 5
    assert len(final.trace) == 5


@pytest.mark.asyncio
async def test_nested_graph_node():
    """Test executing a Graph wrapped inside a NestedGraphNode."""
    # Inner Graph: A -> B
    node_a = TraceNode("InnerA", increment=10)
    node_b = TraceNode("InnerB", increment=20)

    edge = Edge(source="InnerA", target_func=lambda s: "InnerB")

    inner_graph = Graph(
        name="InnerGraph",
        nodes={"InnerA": node_a, "InnerB": node_b},
        edges=[edge],
        entry_point="InnerA",
    )

    nested_node = NestedGraphNode(name="Wrapper", graph=inner_graph)

    # Outer flow could be anything, here we just test the nested node directly
    initial = ExtendedState(counter=0)
    final = await nested_node.execute(initial)

    assert final.counter == 30
    assert final.trace == ["InnerA", "InnerB"]
