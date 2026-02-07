from collections.abc import Callable
from typing import Any

from google.adk.tools.function_tool import FunctionTool

from .agent_tool import AgentAsTool
from .database import DatabaseTool
from .mcp import MCPServerConfig, MCPTool
from .search import GoogleSearchTool


def tool(
    *,
    confirmation: bool | Callable[..., bool] = False,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Wrap a function as an ADK Tool.

    Args:
        confirmation: Whether this tool requires human confirmation.
            Can be a boolean or a callable that takes tool arguments and returns a boolean.

    Example:
        @tool(confirmation=True)
        def sensitive_action(data: str):
            ...

    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, require_confirmation=confirmation)

    return decorator


__all__ = [
    "AgentAsTool",
    "DatabaseTool",
    "GoogleSearchTool",
    "MCPServerConfig",
    "MCPTool",
    "tool",
]
