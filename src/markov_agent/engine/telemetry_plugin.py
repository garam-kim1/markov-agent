from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from markov_agent.core.events import Event, event_bus


class MarkovBridgePlugin(BasePlugin):
    """
    Bridges Google ADK events to the Markov Agent event bus.
    """

    def __init__(self):
        super().__init__(name="markov_bridge")

    async def before_agent_callback(self, *args, **kwargs) -> None:
        await event_bus.emit(Event(name="adk.agent.start", payload={"args": str(args)}))

    async def after_agent_callback(self, *args, **kwargs) -> None:
        await event_bus.emit(Event(name="adk.agent.end", payload={"args": str(args)}))

    async def before_tool_callback(self, tool_name: str, *args, **kwargs) -> None:
        await event_bus.emit(
            Event(name="adk.tool.start", payload={"tool": tool_name, "args": str(args)})
        )

    async def after_tool_callback(
        self, tool_name: str, result: Any, *args, **kwargs
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.tool.end",
                payload={"tool": tool_name, "result": str(result)[:200]},
            )
        )

    async def on_model_error_callback(self, error: Exception, *args, **kwargs) -> None:
        await event_bus.emit(
            Event(name="adk.error", payload={"error": str(error)})
        )
