---
name: adk-tools
description: Expert in designing, implementing, and orchestrating custom tools and toolsets for ADK agents in Python. Use for implementing database queries, API calls, and domain-specific actions.
---

# ADK Tool Architect (Python Edition)

## Philosophy & Architecture
Tools are the "hands" of the agent. In ADK (Python), tools are typically standard Python functions decorated or wrapped to be exposed to the LLM.

## Logic Flow
1. **Define Capabilities**: Identify the discrete actions the agent needs (e.g., `search_web`, `query_db`).
2. **Read References**:
   - For **Function Tools** and **Tool Context**, read `references/tools.md`.
3. **Implementation**:
   - Prefer simple `def` functions with type hints.
   - Use `ToolContext` for state/control flow.
   - Wrap with `FunctionTool` only if advanced configuration (custom schema) is needed.

## Standards
- **Docstrings**: Google-style or Sphinx-style. Must describe *what* the tool does and *what* the arguments are.
- **Type Hints**: Mandatory for all arguments.
- **Return Value**: Must be a `dict` (recommended) or primitive.
- **Async**: `async def` is supported and recommended for I/O bound tools.
