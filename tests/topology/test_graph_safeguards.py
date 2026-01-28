from unittest.mock import MagicMock

import pytest
from google.adk.agents.invocation_context import InvocationContext

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class SimpleState(BaseState):
    count: int = 0

class IncrementNode(BaseNode[SimpleState]):
    async def execute(self, state: SimpleState) -> SimpleState:
        return state.update(count=state.count + 1)

@pytest.mark.asyncio
async def test_graph_max_steps_enforcement():
    """
    Verifies that the graph stops execution when max_steps is reached,
    even if the topology dictates an infinite loop.
    """
    # 1. Setup an Infinite Loop: Node A -> Node A
    node = IncrementNode(name="A")
    
    # Always route back to A
    edge = Edge(source="A", target_func=lambda s: "A")
    
    graph = Graph(
        name="infinite_loop",
        nodes={"A": node},
        edges=[edge],
        entry_point="A",
        max_steps=5, # Limit to 5
        state_type=SimpleState
    )
    
    # 2. Mock ADK Context
    # We need a context that has a session with state
    mock_session = MagicMock()
    # Use a real dict for state so updates work
    state_data = {"count": 0, "history": []}
    mock_session.state = state_data
    
    mock_ctx = MagicMock(spec=InvocationContext)
    mock_ctx.session = mock_session
    mock_ctx.user_content = None # No input injection
    
    # 3. Run the internal async generator
    events = []
    async for event in graph._run_async_impl(mock_ctx):
        events.append(event)
        
    # 4. Verify
    # The node should have executed 5 times.
    # The state count should be 5.
    assert state_data["count"] == 5
    
    # Verify we didn't crash
    # The loop should just exit silently or with a log (we can't easily check log here without caplog)
    # but the count proves it stopped.

@pytest.mark.asyncio
async def test_graph_terminal_node():
    """
    Verifies that execution stops if no edge matches (Terminal Node).
    """
    node = IncrementNode(name="A")
    
    # No edges from A
    
    graph = Graph(
        name="single_step",
        nodes={"A": node},
        edges=[],
        entry_point="A",
        max_steps=10
    )
    
    mock_session = MagicMock()
    state_data = {"count": 0, "history": []}
    mock_session.state = state_data
    
    mock_ctx = MagicMock(spec=InvocationContext)
    mock_ctx.session = mock_session
    mock_ctx.user_content = None
    
    async for _ in graph._run_async_impl(mock_ctx):
        pass
        
    assert state_data["count"] == 1
    # Should stop after 1 step despite max_steps=10

