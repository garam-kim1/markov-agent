from unittest.mock import MagicMock, patch

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


def test_adk_controller_passes_context_configs():
    """Verify that context_cache_config and events_compaction_config
    are passed from ADKConfig to the ADK App.
    """
    # Create mock config objects
    mock_cache_config = MagicMock(name="ContextCacheConfig")
    mock_compaction_config = MagicMock(name="EventsCompactionConfig")

    config = ADKConfig(
        model_name="gemini-1.5-pro",
        context_cache_config=mock_cache_config,
        events_compaction_config=mock_compaction_config,
    )
    retry = RetryPolicy()

    with (
        patch("markov_agent.engine.adk_wrapper.Agent"),
        patch("markov_agent.engine.adk_wrapper.Runner"),
        patch("markov_agent.engine.adk_wrapper.App") as MockApp,
    ):
        ADKController(config, retry)

        MockApp.assert_called_once()
        _, kwargs = MockApp.call_args

        assert kwargs["context_cache_config"] == mock_cache_config
        assert kwargs["events_compaction_config"] == mock_compaction_config


def test_create_variant_preserves_configs():
    """Verify that create_variant preserves the context configurations."""
    mock_cache_config = MagicMock(name="ContextCacheConfig")
    mock_compaction_config = MagicMock(name="EventsCompactionConfig")

    config = ADKConfig(
        model_name="gemini-1.5-pro",
        context_cache_config=mock_cache_config,
        events_compaction_config=mock_compaction_config,
    )
    retry = RetryPolicy()

    # We need to mock the creation of ADKController's internal objects to avoid side effects
    with (
        patch("markov_agent.engine.adk_wrapper.Agent"),
        patch("markov_agent.engine.adk_wrapper.Runner"),
        patch("markov_agent.engine.adk_wrapper.App"),
    ):
        controller = ADKController(config, retry)

        # Create variant
        variant = controller.create_variant({"temperature": 0.5})

        # Check if new config has the context configs
        assert variant.config.context_cache_config == mock_cache_config
        assert variant.config.events_compaction_config == mock_compaction_config
        assert variant.config.temperature == 0.5
