import pytest
from unittest.mock import MagicMock, patch
from markov_agent.tools.mcp import MCPTool, MCPServerConfig

def test_mcp_config_validation():
    # Test missing command for stdio
    config = MCPServerConfig(type="stdio") # command is None by default
    
    with pytest.raises(ValueError, match="Command is required for stdio"):
        MCPTool(config)

def test_mcp_valid_config():
    # Mock McpToolset to avoid actual execution
    with patch("markov_agent.tools.mcp.McpToolset") as MockToolset:
        config = MCPServerConfig(
            type="stdio", 
            command="python", 
            args=["server.py"]
        )
        tool = MCPTool(config)
        
        # Verify toolset initialized
        MockToolset.assert_called_once()
        
        # Verify as_tool_list
        assert len(tool.as_tool_list()) == 1