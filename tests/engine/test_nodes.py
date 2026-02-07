import pytest
from pydantic import BaseModel
from markov_agent.engine.nodes import SearchNode
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.core.state import BaseState

class StateForTest(BaseState):
    query: str
    context: str = ""

def test_search_node_state_type():
    config = ADKConfig(model_name="gemini-1.5-flash")
    # Verify that state_type is correctly passed and stored
    node = SearchNode(
        "searcher", 
        config, 
        "Find {{ query }} in {{ context }}",
        state_type=StateForTest
    )
    
    assert node.state_type == StateForTest
    
    # Test rendering with state_type
    state_dict = {"query": "test query", "context": "test context"}
    prompt = node._render_prompt(state_dict)
    assert "Find test query in test context" in prompt

def test_search_node_tools_injection():
    config = ADKConfig(model_name="gemini-1.5-flash")
    node = SearchNode("searcher", config, "Find {{ query }}")
    
    # GoogleSearchTool.as_tool_list() returns a list containing the tool object
    assert len(node.adk_config.tools) > 0
    # The tool is an instance of AdkGoogleSearchTool or similar
    # In SearchNode it's the result of search_tool_wrapper.as_tool_list()
    # which is [self._tool] where self._tool is AdkGoogleSearchTool
    from google.adk.tools.google_search_tool import GoogleSearchTool as AdkGoogleSearchTool
    assert any(isinstance(t, AdkGoogleSearchTool) for t in node.adk_config.tools)

@pytest.mark.asyncio
async def test_search_node_execute():
    # Use mock responder to avoid actual API calls
    def mock_resp(p):
        return "Search result for test"

    # Use a gemini model name so GoogleSearchTool doesn't complain
    config = ADKConfig(model_name="gemini-1.5-flash")
    node = SearchNode(
        "searcher",
        config,
        "Find {{ query }}",
        mock_responder=mock_resp
    )
    
    state = StateForTest(query="Hello")
    new_state = await node.execute(state)
    
    assert len(new_state.history) > 0
    assert new_state.history[-1]["output"] == "Search result for test"
