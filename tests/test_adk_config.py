import pytest
from unittest.mock import MagicMock, patch
from markov_agent.engine.adk_wrapper import ADKController, ADKConfig, RetryPolicy

def test_adk_controller_initialization():
    """
    Verify that ADKController correctly passes configuration to the underlying ADK Agent.
    """
    config = ADKConfig(
        model_name="gemini-1.5-pro",
        temperature=0.9,
        safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
        api_key="TEST_KEY",
        instruction="Custom instruction",
        description="Custom description"
    )
    retry = RetryPolicy()
    
    # Patch Agent and Runner
    with patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent, \
         patch("markov_agent.engine.adk_wrapper.Runner") as MockRunner, \
         patch.dict("os.environ", {}, clear=True):
        
        controller = ADKController(config, retry)
        
        # Check Environment Variable
        import os
        assert os.environ["GOOGLE_API_KEY"] == "TEST_KEY"
        
        # Check Agent Init
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args[1]
        
        assert call_kwargs["model"] == "gemini-1.5-pro"
        assert call_kwargs["instruction"] == "Custom instruction"
        assert call_kwargs["description"] == "Custom description"
        assert call_kwargs["generate_content_config"]["temperature"] == 0.9
        
        # Check Runner Init
        MockRunner.assert_called_once()
        assert MockRunner.call_args[1]["agent"] == MockAgent.return_value

def test_adk_controller_default_overrides():
    """Verify default instructions and temperature logic."""
    config = ADKConfig(
        model_name="simple-model",
        # temperature defaults to 0.7 in Config, but let's see if it flows
    )
    retry = RetryPolicy()
    
    with patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent, \
         patch("markov_agent.engine.adk_wrapper.Runner"):
        
        controller = ADKController(config, retry)
        
        call_kwargs = MockAgent.call_args[1]
        
        # Check default instruction
        assert "Markov Engine" in call_kwargs["instruction"]
        # Check default temperature
        assert call_kwargs["generate_content_config"]["temperature"] == 0.7
