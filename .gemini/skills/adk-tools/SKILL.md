---
name: adk-tools
description: Expert in designing, implementing, and orchestrating custom tools and toolsets for ADK agents in Python. Use for implementing database queries, API calls, and domain-specific actions.
---

# ADK Tool Architect

## Philosophy & Architecture
Tools are modular code components (functions, classes, or other agents) that an agent calls to perform actions. ADK maps these to model Function Calls.

## Tool Types
1. **Function Tools**: Plain Python functions (`def`) or methods.
2. **Long Running Tools**: Asynchronous or time-consuming operations (`is_long_running=True`).
3. **Agent-as-Tool**: Wrapping another agent as a capability.
4. **`BaseToolset`**: Dynamic grouping of tools based on context.

## Implementation Guidelines
- **Naming**: Descriptive, verb-noun based (e.g., `get_weather`, `cancel_flight`).
- **Arguments**: Must have **type hints** (e.g., `city: str`). JSON-serializable types only.
- **Return Value**: **Must be a `dict`** (or will be automatically wrapped). Use `status: 'success'/'error'`.
- **Docstring**: CRITICAL. The LLM uses this to understand *when* and *how* to call the tool.

## Tool Context Access
- Include `tool_context: ToolContext` in the function signature (not in the docstring).
- Use `tool_context.state` to persist data.
- Use `tool_context.actions` to influence flow:
  - `skip_summarization = True`: Show raw tool output directly.
  - `transfer_to_agent = 'Name'`: Hand off conversation to another agent.
  - `escalate = True`: Signal parent agent or terminate loop.

## Success Criteria
- Valid Python 3.12+ function definitions.
- Detailed docstrings with clear parameter descriptions.
- Correct use of `ToolContext` for state/flow control.
- Read `references/tools.md` for parameter schema.
