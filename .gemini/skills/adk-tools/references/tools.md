# ADK Tools & Toolsets Spec (Python)

## Custom Tool Implementation (`def`)
- **Signature**: `def my_tool(arg1: str, tool_context: ToolContext) -> dict:`
- **Naming**: `verb_noun` (e.g., `get_weather`).
- **Docstring**: CRITICAL. Purpose, arguments, return status ('success'/'error').
- **Return Type**: `dict` (or auto-wrapped).

## Control Patterns (`tool_context.actions`)
- `skip_summarization = True`: Skip the LLM "Final Answer" step.
- `transfer_to_agent = "Name"`: Transfer control.
- `escalate = True`: Terminate loop/signal parent.

## Toolsets (`BaseToolset`)
Dynamic grouping of tools via `get_tools()`.
- **Signature**: `async def get_tools(self, ctx: ReadonlyContext) -> list[BaseTool]:`
- **Use Case**: Role-based access or dynamic configuration.

## Implementation Pattern
```python
def check_urgent(tool_context, query):
    if "urgent" in query:
        tool_context.actions.transfer_to_agent = "support_agent"
    return {"status": "success"}
```

## Parameter Schema
Type hints (e.g., `city: str`) are converted to JSON schema. Use standard serializable types.
- **NO default values** in tool signature.
