from typing import Any, Literal

from google.adk.tools import McpToolset
from mcp import StdioServerParameters
from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for an MCP Server connection.

    Supports both STDIO (local) and HTTP (remote) connections.
    """

    type: Literal["stdio", "http", "sse"] = "stdio"

    # For STDIO
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None

    # For HTTP/SSE
    url: str | None = None
    headers: dict[str, str] | None = None

    # Common
    tool_filter: list[str] | None = None


class MCPTool:
    """A wrapper around Google ADK's McpToolset.

    Enables agents to use tools from Model Context Protocol servers.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._toolset = None
        self._init_toolset()

    def _init_toolset(self) -> None:
        connection_params = None

        if self.config.type == "stdio":
            if not self.config.command:
                msg = "Command is required for stdio MCP server."
                raise ValueError(msg)
            connection_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
        elif self.config.type in {"http", "sse"}:
            # Note: At runtime, we need to import the correct params class
            # This relies on google-adk's internal dependencies or mcp-python
            from mcp.client.sse import (
                SseConnectionParams,  # type: ignore[import-not-found]
            )

            # Assuming basic HTTP/SSE params structure
            connection_params = SseConnectionParams(
                url=self.config.url,
                headers=self.config.headers or {},
            )

        if not connection_params:
            msg = f"Unsupported MCP type: {self.config.type}"
            raise ValueError(msg)

        self._toolset = McpToolset(
            connection_params=connection_params,
            tool_filter=self.config.tool_filter,
        )

    def as_tool_list(self) -> list[Any]:
        """Return the tools managed by McpToolset.

        ADK's McpToolset implements the Toolset interface which likely
        exposes methods or can be passed directly.
        """
        # Based on ADK usage, toolsets are often passed directly to Agent(tools=[...])
        # But if we need a list of functions/tools:
        return [self._toolset]
