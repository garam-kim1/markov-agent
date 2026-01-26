from unittest.mock import patch

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


def test_adk_controller_initialization():
    """
    Verify that ADKController correctly passes configuration to the underlying ADK Agent.
    """
    config = ADKConfig(
        model_name="gemini-1.5-pro",
        temperature=0.9,
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        api_key="TEST_KEY",
        instruction="Custom instruction",
        description="Custom description",
    )
    retry = RetryPolicy()

    # Patch Agent and Runner
    with (
        patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent,
        patch("markov_agent.engine.adk_wrapper.Runner") as MockRunner,
        patch("markov_agent.engine.adk_wrapper.App") as MockApp,
        patch.dict("os.environ", {}, clear=True),
    ):
        ADKController(config, retry)

        MockAgent.assert_called_once()
        _, call_kwargs = MockAgent.call_args

        assert call_kwargs["model"] == "gemini-1.5-pro"
        assert call_kwargs["instruction"] == "Custom instruction"
        assert call_kwargs["description"] == "Custom description"
        assert call_kwargs["generate_content_config"].temperature == 0.9

        # Check Runner Init
        MockRunner.assert_called_once()
        # Ensure App is passed to Runner
        assert MockRunner.call_args[1]["app"] == MockApp.return_value
        
        # Ensure Agent is passed to App
        MockApp.assert_called_once()
        assert MockApp.call_args[1]["root_agent"] == MockAgent.return_value


def test_adk_controller_default_overrides():
    """Verify default instructions and temperature logic."""
    config = ADKConfig(
        model_name="simple-model",
        # temperature defaults to 0.7 in Config, but let's see if it flows
    )
    retry = RetryPolicy()

    with (
        patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent,
        patch("markov_agent.engine.adk_wrapper.Runner"),
        patch("markov_agent.engine.adk_wrapper.App"),
    ):
        ADKController(config, retry)

        args, kwargs = MockAgent.call_args
        # Should default to "You are a probabilistic processing unit..."
