# ADK MCP Spec (Python)

## Roles
1. **MCP Client**: ADK agent consumes tools from an MCP server.
2. **MCP Server**: ADK exposes its tools to other MCP clients.

## Implementation (FastMCP)
ADK uses `FastMCP` for high-level Pythonic server management.
- **Server Setup**:
```python
from fastmcp import FastMCP
mcp = FastMCP("ADK_Server")

@mcp.tool()
async def my_adk_tool(arg: str):
    # ADK logic here
    return {"result": arg}
```

## Consumption Pattern
Integrate MCP tool providers into the `LlmAgent` tools list. ADK handles the protocol mapping.

## Best Practices
- Decorate functions with `@mcp.tool()`.
- Use `FastMCP` for Cloud Run deployment.
- Leverage `mcp-toolbox` for database/API integrations.
