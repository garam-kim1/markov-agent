from typing import Any
from unittest.mock import MagicMock

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext

from markov_agent.engine.callback_adapter import CallbackAdapterPlugin
from markov_agent.engine.callbacks import (
    BeforeAgentCallback,
    BeforeModelCallback,
    BeforeToolCallback,
)


class MockBeforeAgent(BeforeAgentCallback):
    def __init__(self):
        self.called = False

    def __call__(self, context, *args, **kwargs):
        self.called = True


class MockBeforeModel(BeforeModelCallback):
    def __call__(self, context, model_request):
        # Append " - modified" to the request
        return model_request + " - modified"


class MockBeforeTool(BeforeToolCallback):
    def __call__(self, context, tool, tool_args):
        # Add a new arg
        new_args = tool_args.copy()
        new_args["added"] = True
        return new_args


@pytest.mark.asyncio
async def test_callback_adapter_plugin():
    # Setup
    before_agent = MockBeforeAgent()
    before_model = MockBeforeModel()
    before_tool = MockBeforeTool()

    callbacks = [before_agent, before_model, before_tool]
    plugin = CallbackAdapterPlugin(callbacks)

    # Test Before Agent
    ctx = MagicMock(spec=CallbackContext)
    await plugin.before_agent_callback(ctx)
    assert before_agent.called

    # Test Before Model
    req = "request"
    new_req = await plugin.before_model_callback(ctx, req)
    assert new_req == "request - modified"

    # Test Before Tool
    tool_ctx = MagicMock(spec=ToolContext)
    tool_args = {"original": 1}
    new_args = await plugin.before_tool_callback(
        tool=MagicMock(),
        tool_args=tool_args,
        tool_context=tool_ctx,
    )
    assert new_args == {"original": 1, "added": True}


@pytest.mark.asyncio
async def test_multiple_callbacks_chaining():
    class AppendA(BeforeModelCallback):
        def __call__(self, context: CallbackContext, model_request: Any) -> Any:
            return model_request + "A"

    class AppendB(BeforeModelCallback):
        def __call__(self, context: CallbackContext, model_request: Any) -> Any:
            return model_request + "B"

    plugin = CallbackAdapterPlugin([AppendA(), AppendB()])

    # We need a dummy context to avoid type error
    mock_context = MagicMock(spec=CallbackContext)
    req = "start"
    res = await plugin.before_model_callback(mock_context, req)
    assert res == "startAB"
