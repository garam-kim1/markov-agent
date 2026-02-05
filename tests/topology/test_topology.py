import pytest

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


# Define a concrete state for testing
class StateForTest(BaseState):
    value: int = 0
    path: list = []


# Define a concrete node for testing
class AddNode(BaseNode[StateForTest]):
    def __init__(self, name: str, increment: int):
        super().__init__(name)
        self.increment = increment

    async def execute(self, state: StateForTest) -> StateForTest:
        new_value = state.value + self.increment
        new_path = [*state.path, self.name]
        return state.update(value=new_value, path=new_path)


@pytest.mark.asyncio
async def test_simple_graph_execution():
    # Setup nodes
    node_a = AddNode(name="A", increment=1)
    node_b = AddNode(name="B", increment=2)

    # Setup edges
    # A -> B
    def route_a_to_b(state: StateForTest) -> str:
        return "B"

    edge_a = Edge(source="A", target_func=route_a_to_b)

    # Setup graph
    graph = Graph(
        name="test_graph",
        nodes={"A": node_a, "B": node_b},
        edges=[edge_a],
        entry_point="A",
    )

    # Run graph
    initial_state = StateForTest(value=0)
    final_state = await graph.run(initial_state)

    assert final_state.value == 3  # 0 + 1 + 2
    assert final_state.path == ["A", "B"]


@pytest.mark.asyncio
async def test_graph_cycle_limit():
    # Setup nodes
    # A increments by 1
    node_a = AddNode(name="A", increment=1)

    # Setup edges
    # A -> A (Infinite Loop)
    def route_a_to_a(state: StateForTest) -> str:
        return "A"

    edge_a = Edge(source="A", target_func=route_a_to_a)

    graph = Graph(
        name="test_graph",
        nodes={"A": node_a},
        edges=[edge_a],
        entry_point="A",
        max_steps=5,  # Should stop after 5 steps
    )

    initial_state = StateForTest()
    final_state = await graph.run(initial_state)

    # Should have run 5 times
    assert final_state.value == 5
    assert len(final_state.path) == 5


@pytest.mark.asyncio
async def test_conditional_routing():
    # Node A adds 1. If value > 1 goto End, else goto A.
    node_a = AddNode(name="A", increment=1)

    def router(state: StateForTest) -> str:
        if state.value > 2:
            return "END"  # Doesn't exist, so graph terminates
        return "A"

    edge_a = Edge(source="A", target_func=router)

    graph = Graph(
        name="test_graph",
        nodes={"A": node_a},
        edges=[edge_a],
        entry_point="A",
        max_steps=10,
    )

    initial_state = StateForTest()
    final_state = await graph.run(initial_state)

    # 1. Start 0
    # 2. Exec A -> 1. Router: 1 > 2 False -> Goto A
    # 3. Exec A -> 2. Router: 2 > 2 False -> Goto A
    # 4. Exec A -> 3. Router: 3 > 2 True -> Goto END -> Terminate

    assert final_state.value == 3
