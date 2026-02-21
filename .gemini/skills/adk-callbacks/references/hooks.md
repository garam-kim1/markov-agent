# ADK Callbacks & Plugins Spec (Python)

## Plugin Architecture (Global)
Plugins extend `BasePlugin`. Registered in `Runner(plugins=[...])`.
- **Hooks**: `before_run`, `after_run`, `before_agent`, `after_agent`, `before_model`, `after_model`, `before_tool`, `after_tool`, `on_model_error`, `on_tool_error`, `on_event`.
- **Precedence**: Plugin hooks run BEFORE agent-level callbacks.

## Callback Architecture (Local)
Assigned to specific agent instances at creation.
- **Hooks**: `before_agent`, `after_agent`, `before_model`, `after_model`, `before_tool`, `after_tool`.

## Return Logic (Interception)
- `return None`: Proceed as normal.
- `return types.Content` (Agent Hook): Skip agent, use this as final output.
- `return LlmResponse` (Model Hook): Skip LLM call, use this response.
- `return dict` (Tool Hook): Skip tool call, use this as result.

## Context Types
- `InvocationContext`: Full session/service access.
- `CallbackContext`: `(invocation_id, state, agent, services)`.
- `ToolContext`: Adds `function_call_id`.

## Pattern: Guardrail
```python
async def model_guardrail(callback_context, llm_request):
    if "restricted" in llm_request.prompt:
        return LlmResponse(content="Access Denied")
    return None
```
