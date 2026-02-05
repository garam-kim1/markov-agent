from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import ToolContext

from markov_agent.core.events import Event, event_bus


class MarkovBridgePlugin(BasePlugin):
    """Bridges Google ADK events to the Markov Agent event bus."""

    def __init__(self):
        super().__init__(name="markov_telemetry")

    async def before_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.agent.start",
                payload={
                    "agent": callback_context.agent_name,
                    "invocation_id": callback_context.invocation_id,
                    "args": str(args),
                },
            ),
        )

    async def after_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        await event_bus.emit(
            Event(
                name="adk.agent.end",
                payload={
                    "agent": callback_context.agent_name,
                    "invocation_id": callback_context.invocation_id,
                    "args": str(args),
                },
            ),
        )

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        # Note: The order of arguments depends on ADK version.

        await event_bus.emit(
            Event(
                name="adk.tool.start",
                payload={
                    "tool": tool.name if hasattr(tool, "name") else str(tool),
                    "invocation_id": getattr(tool_context, "invocation_id", "unknown"),
                    "function_call_id": getattr(
                        tool_context,
                        "function_call_id",
                        "unknown",
                    ),
                    "args": str(tool_args),
                },
            ),
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        await event_bus.emit(
            Event(
                name="adk.tool.end",
                payload={
                    "tool": tool.name if hasattr(tool, "name") else str(tool),
                    "invocation_id": getattr(tool_context, "invocation_id", "unknown"),
                    "result": str(result)[:200],
                },
            ),
        )
        return None

    async def on_model_error_callback(
        self, error: Exception, *args: Any, **kwargs: Any
    ) -> None:
        await event_bus.emit(Event(name="adk.error", payload={"error": str(error)}))
