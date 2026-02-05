from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import ToolContext

from markov_agent.engine.callbacks import (
    AfterAgentCallback,
    AfterModelCallback,
    AfterToolCallback,
    BeforeAgentCallback,
    BeforeModelCallback,
    BeforeToolCallback,
)


class CallbackAdapterPlugin(BasePlugin):
    """Adapts the Markov Agent Callback interface to the Google ADK Plugin system."""

    def __init__(self, callbacks: list[Any]):
        super().__init__(name="markov_callback_adapter")
        self.callbacks = callbacks

        # Pre-filter callbacks by type for efficiency
        self.before_agent_cbs = [
            cb for cb in callbacks if isinstance(cb, BeforeAgentCallback)
        ]
        self.after_agent_cbs = [
            cb for cb in callbacks if isinstance(cb, AfterAgentCallback)
        ]
        self.before_model_cbs = [
            cb for cb in callbacks if isinstance(cb, BeforeModelCallback)
        ]
        self.after_model_cbs = [
            cb for cb in callbacks if isinstance(cb, AfterModelCallback)
        ]
        self.before_tool_cbs = [
            cb for cb in callbacks if isinstance(cb, BeforeToolCallback)
        ]
        self.after_tool_cbs = [
            cb for cb in callbacks if isinstance(cb, AfterToolCallback)
        ]

    async def before_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for cb in self.before_agent_cbs:
            # Callbacks are defined as sync in the user spec, but we run them here.
            # If we wanted to support async callbacks, we'd check asyncio.iscoroutinefunction
            cb(callback_context, *args, **kwargs)

    async def after_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for cb in self.after_agent_cbs:
            cb(callback_context, *args, **kwargs)

    async def before_model_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # ADK passes 'llm_request' (verified)
        model_request = (
            kwargs.get("llm_request")
            or kwargs.get("model_request")
            or kwargs.get("request")
        )

        if model_request is None and args:
            model_request = args[0]

        if model_request is None:
            return None

        current_request = model_request
        for cb in self.before_model_cbs:
            result = cb(callback_context, current_request)
            if result is not None:
                current_request = result
        return current_request

    async def after_model_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        model_response = (
            kwargs.get("llm_response")
            or kwargs.get("model_response")
            or kwargs.get("response")
            or kwargs.get("result")
        )

        if model_response is None and args:
            model_response = args[0]

        if model_response is None:
            return None

        current_response = model_response
        for cb in self.after_model_cbs:
            result = cb(callback_context, current_response)
            if result is not None:
                current_response = result
        return current_response

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        current_args = tool_args
        modified = False

        for cb in self.before_tool_cbs:
            # Note: User spec says signature is (context, tool, args)
            # ADK plugin gives (tool, args, context)
            # We map to User spec.
            result = cb(tool_context, tool, current_args)
            if result is not None:
                current_args = result
                modified = True

        return current_args if modified else None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        current_result = result
        modified = False

        for cb in self.after_tool_cbs:
            # User spec: (context, tool, args, result)
            res = cb(tool_context, tool, tool_args, current_result)
            if res is not None:
                current_result = res
                modified = True

        return current_result if modified else None
