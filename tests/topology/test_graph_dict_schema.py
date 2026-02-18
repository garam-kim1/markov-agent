from typing import Any

import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.topology.graph import Graph


class SimpleState(BaseState):
    val: str = ""


@pytest.mark.asyncio
async def test_graph_node_with_dict_schema():
    """Test that @graph.node(output_schema=dict) works as expected."""

    def mock_responder(p):
        return '{"new_val": "success", "extra": 123}'

    config = ADKConfig(model_name="mock", mock_responder=mock_responder)

    graph = Graph(name="dict_test", state_type=SimpleState)

    @graph.node(adk_config=config, output_schema=dict)
    def test_node(state: SimpleState, result: dict) -> dict:
        """Update state using a dict result."""
        assert isinstance(result, dict)
        assert result["new_val"] == "success"
        assert result["extra"] == 123
        return {"val": result["new_val"]}

    final_state = await graph.run(SimpleState(val="initial"))

    assert final_state.val == "success"
    # Ensure it's registered
    assert "test_node" in graph.nodes


@pytest.mark.asyncio
async def test_graph_node_with_dict_schema_automatic_merge():
    """Test that @graph.node(output_schema=dict) automatically merges if no state_updater."""

    def mock_responder(p):
        return '{"val": "merged"}'

    config = ADKConfig(model_name="mock", mock_responder=mock_responder)

    graph = Graph(name="merge_test", state_type=SimpleState)

    @graph.node(adk_config=config, output_schema=dict)
    def test_node(state: SimpleState) -> dict:
        """This node doesn't have a state_updater, it should auto-merge result into state."""
        return {}  # Not used by PPU when it has a prompt template

    final_state = await graph.run(SimpleState(val="initial"))

    assert final_state.val == "merged"


@pytest.mark.asyncio
async def test_graph_node_with_dict_schema_invalid_schema_fallback():
    """Test that invalid output_schema (not dict and not BaseModel) defaults to None."""

    graph = Graph(name="fallback_test")

    @graph.node(output_schema=int)  # Not supported, should fallback to None (text)
    def int_node(state: BaseState, result: Any) -> Any:
        return {}

    from markov_agent.engine.ppu import ProbabilisticNode

    node = graph.nodes["int_node"]
    assert isinstance(node, ProbabilisticNode)
    # ProbabilisticNode.output_schema should be None
    assert node.output_schema is None
