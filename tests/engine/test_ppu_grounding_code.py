from google.genai import types

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode


def test_ppu_grounding_config():
    """Test that enabling grounding correctly configures the ADK controller."""
    node = ProbabilisticNode(
        name="grounding_node", prompt_template="Research this.", enable_grounding=True
    )

    assert node.adk_config.enable_grounding is True
    # Verify the tool is added to the agent's tools list
    from google.adk.tools.google_search_tool import GoogleSearchTool

    tools = node.controller.agent.tools
    assert any(isinstance(t, GoogleSearchTool) for t in tools)


def test_ppu_code_execution_config():
    """Test that enabling code execution correctly configures the ADK controller."""
    node = ProbabilisticNode(
        name="coding_node", prompt_template="Write code.", enable_code_execution=True
    )

    assert node.adk_config.enable_code_execution is True
    # Verify the tool is added to the agent's generate_content_config
    gen_config = node.controller.agent.generate_content_config
    assert gen_config is not None
    assert gen_config.tools is not None
    # Inspect tools to find the code execution tool
    found = False
    for t in gen_config.tools:
        if isinstance(t, types.Tool) and t.code_execution is not None:
            found = True
            break
    assert found


def test_ppu_override_config():
    """Test that kwargs override existing ADKConfig."""
    base_config = ADKConfig(model_name="test-model", enable_grounding=False)
    node = ProbabilisticNode(
        name="override_node",
        prompt_template="Override test.",
        adk_config=base_config,
        enable_grounding=True,
    )

    assert node.adk_config.enable_grounding is True
