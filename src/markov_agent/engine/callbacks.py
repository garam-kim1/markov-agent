from abc import ABC, abstractmethod
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext


class CallbackError(Exception):
    """Exception raised by callbacks to stop execution or signal policy violations."""


class BaseCallback(ABC):
    """Base class for all callbacks."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the callback."""


class BeforeAgentCallback(BaseCallback):
    """Callback triggered before the agent starts processing."""

    @abstractmethod
    def __call__(self, context: CallbackContext, *args: Any, **kwargs: Any) -> None:
        pass


class AfterAgentCallback(BaseCallback):
    """Callback triggered after the agent finishes processing."""

    @abstractmethod
    def __call__(self, context: CallbackContext, *args: Any, **kwargs: Any) -> None:
        pass


class BeforeModelCallback(BaseCallback):
    """Callback triggered before sending a request to the model."""

    @abstractmethod
    def __call__(self, context: CallbackContext, model_request: Any) -> Any:
        """Handle the model request before it is sent.

        Args:
            context: The callback context.
            model_request: The request object being sent to the model (e.g. LlmRequest).

        Returns:
            The (potentially modified) model_request, or None to leave it unchanged.

        """


class AfterModelCallback(BaseCallback):
    """Callback triggered after receiving a response from the model."""

    @abstractmethod
    def __call__(self, context: CallbackContext, model_response: Any) -> Any:
        """Handle the model response after it is received.

        Args:
            context: The callback context.
            model_response: The response object received from the model.

        Returns:
            The (potentially modified) model_response, or None to leave it unchanged.

        """


class BeforeToolCallback(BaseCallback):
    """Callback triggered before a tool is executed."""

    @abstractmethod
    def __call__(
        self, context: ToolContext, tool: Any, tool_args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle tool arguments before execution.

        Args:
            context: The tool execution context.
            tool: The tool instance being executed.
            tool_args: The arguments passed to the tool.

        Returns:
            The (potentially modified) tool_args, or None to leave them unchanged.

        """


class AfterToolCallback(BaseCallback):
    """Callback triggered after a tool execution completes."""

    @abstractmethod
    def __call__(
        self, context: ToolContext, tool: Any, tool_args: dict[str, Any], result: Any
    ) -> Any:
        """Handle tool result after execution.

        Args:
            context: The tool execution context.
            tool: The tool instance that was executed.
            tool_args: The arguments passed to the tool.
            result: The result returned by the tool.

        Returns:
            The (potentially modified) result, or None to leave it unchanged.

        """
