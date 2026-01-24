from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import ToolContext

from markov_agent.core.events import Event, event_bus


class MarkovBridgePlugin(BasePlugin):
    """
    Bridges Google ADK events to the Markov Agent event bus.
    """

    def __init__(self):
        super().__init__(name="markov_telemetry")

    async def before_agent_callback(
        self, callback_context: CallbackContext, *args, **kwargs
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.agent.start",
                payload={
                    "agent": callback_context.agent_name,
                    "invocation_id": callback_context.invocation_id,
                    "args": str(args),
                },
            )
        )

    async def after_agent_callback(
        self, callback_context: CallbackContext, *args, **kwargs
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.agent.end",
                payload={
                    "agent": callback_context.agent_name,
                    "invocation_id": callback_context.invocation_id,
                    "args": str(args),
                },
            )
        )

    async def before_tool_callback(
        self, tool_context: ToolContext, tool_name: str, *args, **kwargs
    ) -> None:
        # Note: The order of arguments depends on ADK version.
        # Assuming tool_context is passed first or we handle flexible args if needed.
        # Based on docs: "Passed as tool_context to ... tool execution callbacks"
        # We'll assume typical plugin signature: (context, tool_name, ...)
        
        # If tool_context is not the first arg in some versions, this might need adjustment.
        # But per provided docs, context is primary.
        
        await event_bus.emit(
            Event(
                name="adk.tool.start",
                payload={
                    "tool": tool_name,
                    "invocation_id": getattr(tool_context, "invocation_id", "unknown"),
                    "function_call_id": getattr(
                        tool_context, "function_call_id", "unknown"
                    ),
                    "args": str(args),
                },
            )
        )

    async def after_tool_callback(
        self, tool_context: ToolContext, tool_name: str, result: Any, *args, **kwargs
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.tool.end",
                payload={
                    "tool": tool_name,
                    "invocation_id": getattr(tool_context, "invocation_id", "unknown"),
                    "result": str(result)[:200],
                },
            )
        )

    async def on_model_error_callback(self, error: Exception, *args, **kwargs) -> None:
        await event_bus.emit(Event(name="adk.error", payload={"error": str(error)}))
