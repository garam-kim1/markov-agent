import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


@pytest.mark.asyncio
async def test_adk_controller_enables_logging():
    config = ADKConfig(model_name="gemini-1.5-flash", enable_logging=True)

    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))

    # Check if LoggingPlugin is in the app's plugins
    plugin_names = [p.name for p in controller.app.plugins]
    assert "logging_plugin" in plugin_names


@pytest.mark.asyncio
async def test_adk_controller_enables_tracing(monkeypatch):
    # Mock configure_local_telemetry to avoid setting global provider in tests
    called = False

    def mock_configure():
        nonlocal called
        called = True

    monkeypatch.setattr(
        "markov_agent.engine.observability.configure_local_telemetry", mock_configure
    )

    config = ADKConfig(model_name="gemini-1.5-flash", enable_tracing=True)

    ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))

    assert called is True


@pytest.mark.asyncio
async def test_observability_functional_flow():
    """Verify full flow with MockLLM and observability enabled."""
    from collections.abc import AsyncGenerator
    from typing import Any

    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class MockLLM(BaseLlm):
        model: str = "mock-test-model"

        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="Mock Response")]
                ),
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=5, candidates_token_count=5, total_token_count=10
                ),
                finish_reason=types.FinishReason.STOP,
            )

    config = ADKConfig(
        model_name=MockLLM(),
        enable_logging=True,
        enable_tracing=False,  # Disable tracing for this test to avoid global state issues
    )

    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))
    response = await controller.generate("test prompt")
    assert response == "Mock Response"

    # Verify LoggingPlugin is present
    assert any(p.name == "logging_plugin" for p in controller.app.plugins)
