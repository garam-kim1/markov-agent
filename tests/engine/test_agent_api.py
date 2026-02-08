import pytest
from markov_agent import Agent, model_config, ADKConfig

def test_model_config_helper():
    config = model_config("gemini-1.5-flash", temperature=0.5, top_p=0.9)
    assert isinstance(config, ADKConfig)
    assert config.model_name == "gemini-1.5-flash"
    assert config.temperature == 0.5
    assert config.top_p == 0.9

def test_agent_initialization():
    agent = Agent(
        name="test_agent",
        model="gemini-1.5-flash",
        system_prompt="You are a tester."
    )
    assert agent.name == "test_agent"
    assert agent.adk_config.model_name == "gemini-1.5-flash"
    assert agent.system_prompt == "You are a tester."

def test_agent_add_tool():
    def mock_tool(x: int) -> int:
        return x + 1
    
    agent = Agent(name="test_agent")
    agent.add_tool(mock_tool)
    assert mock_tool in agent.adk_config.tools

@pytest.mark.asyncio
async def test_agent_run_mock():
    # We use a mock_responder to test the run method without API calls
    def mock_responder(prompt: str) -> str:
        return f"Mock response to: {prompt}"
    
    agent = Agent(
        name="test_agent",
        mock_responder=mock_responder
    )
    
    # Test sync run
    response = agent.run("Hello")
    assert response.text == "Mock response to: Hello"
    assert str(response) == "Mock response to: Hello"

    # Test async run
    response_text = await agent.run_async_text("Hello Async")
    assert response_text == "Mock response to: Hello Async"

def test_agent_system_prompt_setter():
    agent = Agent(name="test_agent", system_prompt="Initial")
    assert agent.system_prompt == "Initial"
    
    agent.system_prompt = "Updated"
    assert agent.system_prompt == "Updated"
    assert agent.adk_config.instruction == "Updated"
