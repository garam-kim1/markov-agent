---
name: adk-mcp
description: Expert in Model Context Protocol (MCP) integration with ADK agents in Python. Use for implementing universal connections between LLMs and external data sources or toolsets.
---

# ADK MCP Specialist

## Philosophy & Architecture
MCP is an open standard to standardize LLM-tool communication. ADK can act as an MCP client or host an MCP server. This project uses the official `mcp` Python SDK.

## MCP in ADK
1. **Consuming Tools**: ADK acts as an MCP client and uses tools provided by external MCP servers.
2. **Exposing Tools**: Build an MCP server that wraps ADK tools using the `mcp` SDK.

## Implementation
- **mcp SDK**: Use the official SDK to handle protocol and server management.
- **Client**: Connect to servers (e.g., databases, app-specific APIs) and map them to `BaseTool` instances.

## Best Practices
- Use the `mcp` SDK for Python-based servers.
- For Markov Agents, MCP tools can be integrated into `FunctionalNode` or as external capabilities for `ProbabilisticNode`.
- Decorate functions to expose them as MCP tools.
- Read `references/mcp.md` for server setup and Cloud Run deployment.

## Success Criteria
- Valid MCP server definitions.
- Successful tool mapping between MCP and ADK.
- Robust handling of tool-specific MCP contexts.
