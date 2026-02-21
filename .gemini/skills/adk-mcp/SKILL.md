---
name: adk-mcp
description: Expert in Model Context Protocol (MCP) integration with ADK agents in Python. Use for implementing universal connections between LLMs and external data sources or toolsets.
---

# ADK MCP Specialist (Python Edition)

## Philosophy & Architecture
MCP is an open standard to standardize LLM-tool communication. ADK can act as an MCP client or host an MCP server. This project uses the official `mcp` Python SDK.

## MCP in ADK
1. **Consuming Tools (Client)**:
   - Use `McpToolset` to connect to local (stdio) or remote (SSE) MCP servers.
   - Tools are dynamically loaded and exposed to the `LlmAgent`.
2. **Exposing Tools (Server)**:
   - Build an MCP server using `FastMCP` or the `mcp` SDK to wrap Python functions.

## Implementation
- **mcp SDK**: Use the official SDK to handle protocol management.
- **Client**: Connect to servers (e.g., databases, app-specific APIs).
- Read `references/mcp.md` for server setup and tool consumption patterns.

## Best Practices
- Use `FastMCP` for rapid Python-based server development.
- Decorate functions to expose them as MCP tools.
- Validate sensitive MCP tool calls using ADK `before_tool` callbacks.

## Success Criteria
- Valid `McpToolset` configuration connecting to a running server.
- Successful execution of remote MCP tools by the agent.
- Secure handling of tool permissions.
