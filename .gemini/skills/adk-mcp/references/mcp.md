# ADK MCP Spec (Python)

## 1. Architecture
- **MCP Client**: ADK agents consuming external tools.
- **MCP Server**: Python services exposing tools via standard protocol.

## 2. Consuming Tools (Client Mode)
ADK provides `McpToolset` to connect to running MCP servers (stdio or SSE).

```python
from google.adk.tools import McpToolset

# Connect to a local MCP server running via stdio
mcp_tools = McpToolset(
    command="uv",
    args=["run", "mcp-server-sqlite", "--db-path", "./test.db"]
)

agent = LlmAgent(
    model="gemini-2.0-flash",
    tools=[mcp_tools], # Dynamically loads tools from server
    instruction="Query the database for user details."
)
```

## 3. Building Servers (Server Mode)
Use the official `mcp` SDK or `FastMCP` (if available) to build servers.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyADKService")

@mcp.tool()
def calculate_metrics(data: list[float]) -> dict:
    """Calculates statistical metrics."""
    return {"mean": sum(data)/len(data)}

# Run: uv run mcp run server.py
```

## 4. Security
- **Authorization**: MCP servers run with the privileges of the host process.
- **Human-in-the-loop**: Use `before_tool` callbacks in ADK to approve sensitive MCP tool calls.
