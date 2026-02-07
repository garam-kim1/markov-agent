from collections.abc import Callable
from typing import Any, overload

from google.adk.tools.function_tool import FunctionTool

from .agent_tool import AgentAsTool
from .database import DatabaseTool
from .mcp import McpClient, MCPServerConfig, MCPTool, SseMcpTransport, StdioMcpTransport
from .search import GoogleSearchTool


@overload
def tool(func: Callable[..., Any]) -> FunctionTool: ...


@overload
def tool(
    *,
    confirmation: bool | Callable[..., bool] = False,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    confirmation: bool | Callable[..., bool] = False,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Wrap a function as an ADK Tool.

    Args:
        func: The function to wrap.
        confirmation: Whether this tool requires human confirmation.
            Can be a boolean or a callable that takes tool arguments and returns a boolean.

    Example:
        @tool
        def basic_tool(arg: str):
            ...

        @tool(confirmation=True)
        def sensitive_action(data: str):
            ...

    """

    def decorator(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, require_confirmation=confirmation)

    if func is None:
        return decorator
    return decorator(func)


__all__ = [
    "AgentAsTool",
    "DatabaseTool",
    "GoogleSearchTool",
    "MCPServerConfig",
    "MCPTool",
    "McpClient",
    "SseMcpTransport",
    "StdioMcpTransport",
    "tool",
]
