from unittest.mock import patch

import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


def test_extended_config_google_mapping():
    """
    Test that parameters are mapped correctly for Google GenAI (default).
    Checks mapping of max_tokens -> max_output_tokens.
    Ensures unsupported params like min_p are dropped from GenerateContentConfig.
    """
    config = ADKConfig(
        model_name="gemini-1.5-pro",
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        max_tokens=1000,
        min_p=0.05
    )
    retry = RetryPolicy()

    with (
        patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent,
        patch("markov_agent.engine.adk_wrapper.Runner"),
        patch.dict("os.environ", {}, clear=True),
    ):
        try:
            ADKController(config, retry)
        except Exception as e:
            pytest.fail(f"ADKController initialization failed: {e}")

        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args[1]
        
        gen_config = call_kwargs["generate_content_config"]
        
        # Verify supported fields
        assert gen_config.temperature == 0.8
        assert gen_config.top_p == 0.95
        assert gen_config.top_k == 40
        assert gen_config.frequency_penalty == 0.1
        assert gen_config.presence_penalty == 0.2
        assert gen_config.max_output_tokens == 1000
        
        # Verify unsupported fields (min_p) are NOT present (dropped)
        assert not hasattr(gen_config, "min_p")
        # Ensure max_tokens is gone
        assert not hasattr(gen_config, "max_tokens")


def test_extended_config_litellm_mapping():
    """
    Test that parameters are mapped correctly when use_litellm=True.
    Checks that max_tokens -> max_output_tokens (mapped for safe config).
    Checks that min_p is passed to LiteLlm constructor via extra_kwargs.
    """
    config = ADKConfig(
        model_name="openai/gpt-4o",
        use_litellm=True,
        max_tokens=500,
        min_p=0.1,
        temperature=0.7
    )
    retry = RetryPolicy()

    with (
        patch("markov_agent.engine.adk_wrapper.Agent") as MockAgent,
        patch("markov_agent.engine.adk_wrapper.Runner"),
        patch.dict("os.environ", {}, clear=True),
    ):
        # Patch the source module class
        with patch("google.adk.models.lite_llm.LiteLlm") as MockLiteLLMClass:
            ADKController(config, retry)
            
            # Check LiteLlm init args
            MockLiteLLMClass.assert_called_once()
            _, kwargs = MockLiteLLMClass.call_args
            assert kwargs["model"] == "openai/gpt-4o"
            assert kwargs["min_p"] == 0.1  # min_p passed here!
            
            # Check Agent init
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            gen_config = call_kwargs["generate_content_config"]
            
            # max_tokens mapped to max_output_tokens for GenerateContentConfig
            assert gen_config.max_output_tokens == 500
            assert gen_config.temperature == 0.7
            
            # min_p should NOT be in GenerateContentConfig
            assert not hasattr(gen_config, "min_p")


def test_create_variant_overrides():
    """
    Test that create_variant correctly updates the new parameters.
    """
    config = ADKConfig(
        model_name="test",
        temperature=0.5,
        top_p=0.9
    )
    retry = RetryPolicy()
    
    with (
        patch("markov_agent.engine.adk_wrapper.Agent"),
        patch("markov_agent.engine.adk_wrapper.Runner"),
    ):
        controller = ADKController(config, retry)
        
        # Create variant
        variant = controller.create_variant(
            {"temperature": 0.1, "top_p": 0.5, "top_k": 10}
        )
        
        # Check variant config
        assert variant.config.temperature == 0.1
        assert variant.config.top_p == 0.5
        assert variant.config.top_k == 10
        assert variant.config.model_name == "test"
        
        # Verify that original controller is unchanged
        assert controller.config.temperature == 0.5
