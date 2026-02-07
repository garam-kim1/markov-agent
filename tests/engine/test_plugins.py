from unittest.mock import MagicMock

import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.plugins import (
    BasePlugin,
    CallbackContext,
    LlmRequest,
)


class MockPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="mock_plugin")
        self.before_agent_called = False
        self.before_model_called = False

    async def before_agent_callback(self, *, agent, callback_context: CallbackContext):
        self.before_agent_called = True
        return

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ):
        self.before_model_called = True
        return


@pytest.mark.asyncio
async def test_adk_controller_with_custom_plugin():
    # Setup
    plugin = MockPlugin()
    config = ADKConfig(model_name="gemini-3-flash-preview", plugins=[plugin])
    retry_policy = RetryPolicy()

    # We mock the runner to avoid actual LLM calls
    controller = ADKController(config=config, retry_policy=retry_policy)

    # Mock the runner's run_async method
    controller.runner.run_async = MagicMock()  # type: ignore

    # In a real scenario, ADK would call the plugin hooks.
    # Since we are mocking the runner, we can't easily trigger the hooks through it
    # without deeper mocking of ADK internals.

    # However, we can verify that the plugin is in the app
    assert plugin in controller.app.plugins
    assert controller.app.plugins[-1] == plugin


def test_plugin_exports():
    from markov_agent import (
        BasePlugin,
        BaseTool,
        CallbackContext,
        LlmRequest,
        LlmResponse,
        ToolContext,
        types,
    )

    assert BasePlugin is not None
    assert CallbackContext is not None
    assert ToolContext is not None
    assert BaseTool is not None
    assert LlmRequest is not None
    assert LlmResponse is not None
    assert types is not None
