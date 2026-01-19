import pytest
from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode
from google.adk.agents.invocation_context import InvocationContext

# Define a concrete state for testing
class AdvancedState(BaseState):
    value: int = 0
    path: list[str] = []

# Define a concrete node for testing
class SimpleNode(BaseNode[AdvancedState]):
    def __init__(self, name: str, increment: int = 1):
        super().__init__(name)
        self.increment = increment

    async def execute(self, state: AdvancedState) -> AdvancedState:
        new_value = state.value + self.increment
        new_path = state.path + [self.name]
        return state.update(value=new_value, path=new_path)

class ErrorNode(BaseNode[AdvancedState]):
    async def execute(self, state: AdvancedState) -> AdvancedState:
        raise ValueError("Intentional Failure")

@pytest.mark.asyncio
async def test_invalid_transition():
    """Test when an edge points to a non-existent node ID."""
    node_a = SimpleNode(name="A")
    
    # Route to non-existent node "Z"
    def route_to_z(state: AdvancedState) -> str:
        return "Z"
    
    edge_a = Edge(source="A", target_func=route_to_z)
    
    graph = Graph(
        name="test_graph_invalid",
        nodes={"A": node_a},
        edges=[edge_a],
        entry_point="A"
    )
    
    initial_state = AdvancedState()
    final_state = await graph.run(initial_state)
    
    # Should execute A, try to go to Z, fail to find Z, and terminate gracefully
    assert final_state.path == ["A"]
    assert final_state.value == 1

@pytest.mark.asyncio
async def test_node_exception():
    """Test behavior when a node raises an exception."""
    node_error = ErrorNode(name="ErrorNode")
    
    graph = Graph(
        name="test_graph_error",
        nodes={"ErrorNode": node_error},
        edges=[],
        entry_point="ErrorNode"
    )
    
    initial_state = AdvancedState()
    
    # Expect the exception to propagate out of graph.run
    with pytest.raises(ValueError, match="Intentional Failure"):
        await graph.run(initial_state)

@pytest.mark.asyncio
async def test_disconnected_entry():
    """Test a graph where the entry point has no outgoing edges."""
    node_a = SimpleNode(name="A")
    
    # No edges provided
    graph = Graph(
        name="test_graph_disconnected",
        nodes={"A": node_a},
        edges=[],
        entry_point="A"
    )
    
    initial_state = AdvancedState()
    final_state = await graph.run(initial_state)
    
    # Should execute A and then stop
    assert final_state.path == ["A"]
    assert final_state.value == 1

@pytest.mark.asyncio
async def test_state_persistence_in_loop():
    """Verify that state updates persist across multiple nodes."""
    node_a = SimpleNode(name="A", increment=10)
    node_b = SimpleNode(name="B", increment=5)
    
    # A -> B
    edge_a = Edge(source="A", target_func=lambda s: "B")
    
    graph = Graph(
        name="test_graph_persistence",
        nodes={"A": node_a, "B": node_b},
        edges=[edge_a],
        entry_point="A"
    )
    
    initial_state = AdvancedState(value=100)
    final_state = await graph.run(initial_state)
    
    # 100 + 10 (A) + 5 (B) = 115
    assert final_state.value == 115
    assert final_state.path == ["A", "B"]
