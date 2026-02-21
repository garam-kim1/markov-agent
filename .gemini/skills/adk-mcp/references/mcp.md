# ADK MCP Spec (Python)

## Roles
1. **MCP Client**: ADK agent consumes tools from an MCP server.
2. **MCP Server**: ADK exposes its tools to other MCP clients.

## Implementation (mcp SDK)
ADK uses the official `mcp` SDK for Pythonic server management.

### Server Setup (mcp)
```python
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("ADK_Server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_adk_tool",
            description="An ADK tool exposed via MCP",
            inputSchema={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"]
            }
        )
    ]
```

## Consumption Pattern
Integrate MCP tool providers into the `LlmAgent` tools list. ADK handles the protocol mapping via `mcp.client`.

## Markov Agent Integration
MCP servers can be used to provide real-time data to `ProbabilisticNode` or as external state updates in `FunctionalNode`.
