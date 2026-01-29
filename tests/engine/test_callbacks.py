import pytest
from unittest.mock import Mock, MagicMock
from markov_agent.engine.callbacks import (
    BeforeAgentCallback,
    AfterAgentCallback,
    BeforeModelCallback,
    AfterModelCallback,
    BeforeToolCallback,
    AfterToolCallback,
)
from markov_agent.engine.callback_adapter import CallbackAdapterPlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext


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
        tool_context=tool_ctx
    )
    assert new_args == {"original": 1, "added": True}

@pytest.mark.asyncio
async def test_multiple_callbacks_chaining():
    class AppendA(BeforeModelCallback):
        def __call__(self, context, req):
            return req + "A"
            
    class AppendB(BeforeModelCallback):
        def __call__(self, context, req):
            return req + "B"
            
    plugin = CallbackAdapterPlugin([AppendA(), AppendB()])
    
    req = "start"
    res = await plugin.before_model_callback(None, req)
    assert res == "startAB"
