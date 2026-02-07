from unittest.mock import patch

import pytest

from markov_agent.tools.mcp import (
    McpClient,
    MCPServerConfig,
    MCPTool,
    SseMcpTransport,
    StdioMcpTransport,
)


def test_mcp_config_validation():
    # Test missing command for stdio
    config = MCPServerConfig(type="stdio")  # command is None by default

    with pytest.raises(ValueError, match="Command is required for stdio"):
        MCPTool(config)


def test_mcp_valid_config():
    # Mock McpToolset to avoid actual execution
    with patch("markov_agent.tools.mcp.McpToolset") as MockToolset:
        config = MCPServerConfig(type="stdio", command="python", args=["server.py"])
        tool = MCPTool(config)

        # Verify toolset initialized
        MockToolset.assert_called_once()

        # Verify as_tool_list
        assert len(tool.as_tool_list()) == 1


def test_mcp_client_and_transports():
    with patch("markov_agent.tools.mcp.McpToolset") as MockToolset:
        # Test Stdio Transport
        stdio_transport = StdioMcpTransport(command="python", args=["server.py"])
        client = McpClient(transport=stdio_transport)
        assert client.transport == stdio_transport
        MockToolset.assert_called_once()

        # Test SSE Transport
        MockToolset.reset_mock()
        sse_transport = SseMcpTransport(url="http://localhost:8000/sse")
        client = McpClient(transport=sse_transport)
        assert client.transport == sse_transport
        MockToolset.assert_called_once()
