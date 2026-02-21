# ADK Plugins Spec (Python)

## Plugin Implementation
Subclass `google.adk.plugins.BasePlugin`.
- `name` (str): Unique ID.
- `callback_hooks` (list): Hook method names.

## Plugin Hooks (Global)
1. **Runner Start**: `before_run_callback` / `after_run_callback`.
2. **User Input**: `on_user_message_callback`.
3. **Agent Lifecycle**: `before_agent_callback` / `after_agent_callback`.
4. **Model Lifecycle**: `before_model_callback` / `after_model_callback` / `on_model_error_callback`.
5. **Tool Lifecycle**: `before_tool_callback` / `after_tool_callback` / `on_tool_error_callback`.
6. **Event Processing**: `on_event_callback`.

## Execution Patterns
- **Observe**: Implement hook with `return None`.
- **Intervene**: Implement hook and `return <Specific Object>` (e.g., `LlmResponse`, `types.Content`, `dict`).
- **Amend**: Modify `invocation_context` or `callback_context` properties directly.

## Success Criteria
- Global enforcement of policies (e.g., security, caching, logging).
- Correct registration in `Runner(plugins=[...])`.
- Precedence over local callbacks correctly handled.
