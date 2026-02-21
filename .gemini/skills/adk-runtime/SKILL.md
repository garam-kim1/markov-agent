---
name: adk-runtime
description: Expert in ADK agent runtime configuration, execution modes, and server operations in Python. Use for configuring the Dev UI, CLI runs, and RESTful API servers.
---

# ADK Runtime Specialist

## Philosophy & Architecture
The runtime powers the execution of agents. ADK provides interactive and non-interactive ways to run agents.

## Running Agents
1. **Dev UI (`adk web`)**: Browser-based interface for interaction and debugging.
2. **Command Line (`adk run`)**: Terminal-based interaction.
3. **API Server (`adk api_server`)**: Expose agents through a RESTful API.

## Technical Concepts
- **Event Loop**: Understand the yield/pause/resume cycle.
- **Session Resumption**: Learn how to resume execution from a previous state.
- **`RunConfig`**: Global configuration for runtime behavior.

## Success Criteria
- Valid configuration of `RunConfig`.
- Successful launching of the API server.
- Proper handling of the event stream from the `Runner`.
- Read `references/runtime.md` for CLI command details.
