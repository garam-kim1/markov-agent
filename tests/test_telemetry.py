import pytest
from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext

from markov_agent.core.events import Event, event_bus
from markov_agent.engine.telemetry_plugin import MarkovBridgePlugin


@pytest.mark.asyncio
async def test_telemetry_plugin_emits_events():
    # 1. Setup subscriber
    received_events = []

    async def capture_event(event: Event):
        received_events.append(event)

    # Subscribe to all ADK events
    event_bus.subscribe("adk.agent.start", capture_event)
    event_bus.subscribe("adk.tool.end", capture_event)
    event_bus.subscribe("adk.error", capture_event)

    plugin = MarkovBridgePlugin()

    # 2. Trigger Plugin Methods with Mocks
    mock_agent_ctx = MagicMock(spec=CallbackContext)
    mock_agent_ctx.agent_name = "test_agent"
    mock_agent_ctx.invocation_id = "inv-123"

    mock_tool_ctx = MagicMock(spec=ToolContext)
    mock_tool_ctx.invocation_id = "inv-123"
    
    await plugin.before_agent_callback(mock_agent_ctx, "arg1", kw="arg2")
    
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    
    # New signature: after_tool_callback(*, tool, tool_args, tool_context, result)
    await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args={"some": "arg"},
        tool_context=mock_tool_ctx,
        result={"output": "result_data"}
    )
    
    await plugin.on_model_error_callback(ValueError("Test Error"))

    # 3. Assertions
    assert len(received_events) == 3

    assert received_events[0].name == "adk.agent.start"
    assert received_events[0].payload["agent"] == "test_agent"
    assert received_events[0].payload["invocation_id"] == "inv-123"
    assert "arg1" in received_events[0].payload["args"]

    assert received_events[1].name == "adk.tool.end"
    assert received_events[1].payload["tool"] == "test_tool"
    assert "result_data" in received_events[1].payload["result"]
    assert received_events[1].payload["invocation_id"] == "inv-123"

    assert received_events[2].name == "adk.error"
    assert "Test Error" in received_events[2].payload["error"]
