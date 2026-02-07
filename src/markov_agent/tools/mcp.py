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


class StdioMcpTransport:
    """Transport configuration for local stdio MCP servers."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env

    def to_adk_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )


class SseMcpTransport:
    """Transport configuration for remote SSE MCP servers."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.url = url
        self.headers = headers or {}

    def to_adk_params(self) -> Any:
        from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

        return SseConnectionParams(
            url=self.url,
            headers=self.headers,
        )


class McpClient:
    """Client for connecting to an MCP server and discovering tools."""

    def __init__(
        self,
        transport: StdioMcpTransport | SseMcpTransport,
        tool_filter: list[str] | None = None,
    ):
        self.transport = transport
        self.tool_filter = tool_filter
        self._toolset = McpToolset(
            connection_params=transport.to_adk_params(),
            tool_filter=tool_filter,
        )

    def as_tool_list(self) -> list[Any]:
        return [self._toolset]


class MCPTool:
    """A wrapper around Google ADK's McpToolset.

    Enables agents to use tools from Model Context Protocol servers.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._toolset = None
        self._init_toolset()

    def _init_toolset(self) -> None:
        if self.config.type == "stdio":
            if not self.config.command:
                msg = "Command is required for stdio MCP server."
                raise ValueError(msg)
            transport = StdioMcpTransport(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env,
            )
        elif self.config.type in {"http", "sse"}:
            if not self.config.url:
                msg = "URL is required for SSE MCP server."
                raise ValueError(msg)
            transport = SseMcpTransport(
                url=self.config.url,
                headers=self.config.headers,
            )
        else:
            msg = f"Unsupported MCP type: {self.config.type}"
            raise ValueError(msg)

        self._client = McpClient(
            transport=transport, tool_filter=self.config.tool_filter
        )
        self._toolset = self._client._toolset

    def as_tool_list(self) -> list[Any]:
        """Return the tools managed by McpToolset."""
        return [self._toolset]
