import pytest
from markov_agent.engine.telemetry_plugin import MarkovBridgePlugin
from markov_agent.core.events import event_bus, Event

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
    
    # 2. Trigger Plugin Methods
    await plugin.before_agent_callback("arg1", kw="arg2")
    await plugin.after_tool_callback("test_tool", "result_data")
    await plugin.on_model_error_callback(ValueError("Test Error"))
    
    # 3. Assertions
    # We might need to wait briefly if event bus is async-detached, 
    # but the implementation uses asyncio.gather so it should be awaited.
    
    assert len(received_events) == 3
    
    assert received_events[0].name == "adk.agent.start"
    assert "arg1" in received_events[0].payload["args"]
    
    assert received_events[1].name == "adk.tool.end"
    assert received_events[1].payload["tool"] == "test_tool"
    assert received_events[1].payload["result"] == "result_data"
    
    assert received_events[2].name == "adk.error"
    assert "Test Error" in received_events[2].payload["error"]
