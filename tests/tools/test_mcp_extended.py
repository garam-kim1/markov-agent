from unittest.mock import patch

from markov_agent.tools.mcp import MCPServerConfig, MCPTool


def test_mcp_http_config():
    """Test that HTTP/SSE configuration correctly initializes SseConnectionParams."""
    # Mock SseConnectionParams in the location where it's now imported from
    with (
        patch(
            "google.adk.tools.mcp_tool.mcp_session_manager.SseConnectionParams"
        ) as mock_params_class,
        patch("markov_agent.tools.mcp.McpToolset") as MockToolset,
    ):
        config = MCPServerConfig(
            type="sse",
            url="http://localhost:8080/sse",
            headers={"Authorization": "Bearer token"},
        )

        _ = MCPTool(config)

        # Verify SseConnectionParams was initialized with correct args
        mock_params_class.assert_called_once_with(
            url="http://localhost:8080/sse",
            headers={"Authorization": "Bearer token"},
        )

        # Verify toolset initialized with these params
        MockToolset.assert_called_once()
        _, kwargs = MockToolset.call_args
        assert kwargs["connection_params"] == mock_params_class.return_value


def test_mcp_unsupported_type_fallback():
    """Test logic fall through if type is technically valid by Pydantic but not handled in if/else
    (though currently covered by Pydantic enum, good to be safe if enum expands).
    """
    # Create a config that bypasses Pydantic validation for the 'type' field to test runtime check
    # or just subclass it.

    # Actually, simpler: construct config with valid type but fail the params creation
    # But the code logic is strict.
