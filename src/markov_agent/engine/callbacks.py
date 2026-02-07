import re
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


class SafetyGuardrail(BeforeModelCallback):
    """A guardrail that inspects user input for unsafe content before it is sent to the model.

    If a violation is detected, raises CallbackError to halt execution.
    """

    def __init__(self, blocked_terms: list[str]):
        self.blocked_terms = [term.lower() for term in blocked_terms]

    def __call__(self, context: CallbackContext, model_request: Any) -> None:
        """Intercept the prompt. Raise CallbackError if blocked terms are found.

        Args:
            context: The callback context.
            model_request: The request object. We expect it to have a 'contents' attribute
                or similar containing the prompts.

        """
        # In ADK, model_request is often an LlmRequest.
        # We try to find the prompt text in its contents.
        prompts = []
        if hasattr(model_request, "contents"):
            for content in model_request.contents:
                if hasattr(content, "parts"):
                    prompts.extend(
                        [
                            part.text
                            for part in content.parts
                            if hasattr(part, "text") and part.text
                        ]
                    )
        elif isinstance(model_request, str):
            prompts = [model_request]

        for prompt in prompts:
            lowered_prompt = prompt.lower()
            for term in self.blocked_terms:
                if term in lowered_prompt:
                    msg = (
                        f"Safety Violation: Prompt contains blocked content ('{term}')."
                    )
                    raise CallbackError(msg)


class PIIRedactionCallback(BeforeModelCallback):
    """A callback that redacts PII (Emails, Phone Numbers) from the prompt.

    Demonstrates layered defense.
    """

    EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    PHONE_REGEX = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

    def __call__(self, context: CallbackContext, model_request: Any) -> Any:
        """Redacts PII from the model request."""
        if not hasattr(model_request, "contents"):
            return None

        for content in model_request.contents:
            if hasattr(content, "parts"):
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        original_text = part.text
                        new_text = self.EMAIL_REGEX.sub(
                            "[EMAIL_REDACTED]", original_text
                        )
                        new_text = self.PHONE_REGEX.sub("[PHONE_REDACTED]", new_text)

                        if new_text != original_text:
                            part.text = new_text

        return None
