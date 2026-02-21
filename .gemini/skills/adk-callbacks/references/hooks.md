# ADK Callbacks & Plugins Spec (Python)

## 1. Distinction: Plugin vs. Callback
| Feature | **Plugin** | **Callback** |
| :--- | :--- | :--- |
| **Scope** | Global (Runner-level) | Local (Agent/Tool-level) |
| **Registration** | `Runner(plugins=[...])` | `Agent(..., before_agent_callback=...)` |
| **Use Case** | Logging, Metrics, Global Security | Validation, State Logic, Agent Hand-off |
| **Precedence** | Runs **BEFORE** callbacks | Runs **AFTER** plugins |

## 2. Available Hooks
Both Plugins and Callbacks use the same hook signatures.

| Hook Name | Trigger Point | Return to Intercept |
| :--- | :--- | :--- |
| `before_agent` | Before agent logic starts | `types.Content` (Final Response) |
| `after_agent` | After agent finishes | `types.Content` (Modify Output) |
| `before_model` | Before LLM API call | `LlmResponse` (Mock/Cache) |
| `after_model` | After LLM response | `LlmResponse` (Modify Generation) |
| `before_tool` | Before tool execution | `dict` (Mock Result/Block) |
| `after_tool` | After tool execution | `dict` (Modify Tool Output) |
| `on_event` | On any system event | `None` (Observability only) |

## 3. Implementation Pattern (Callback)
Callbacks are simple functions.

```python
def my_before_tool_callback(callback_context, tool, args, tool_context):
    print(f"Tool {tool.name} called with {args}")
    
    # Example: Block execution based on logic
    if args.get("dry_run") is True:
        return {"status": "dry_run_success"}
    
    return None # Continue execution
```

## 4. Implementation Pattern (Plugin)
Plugins inherit from `BasePlugin`.

```python
from google.adk.plugins import BasePlugin

class PIIFilterPlugin(BasePlugin):
    def before_model(self, callback_context, llm_request):
        # Redact sensitive info from prompt
        llm_request.prompt = redact_pii(llm_request.prompt)
        return None
```

## 5. Context Objects
- **`InvocationContext`**: Full access to `session`, `user_id`, `state`.
- **`CallbackContext`**: Lightweight wrapper passed to hooks.
- **`ToolContext`**: Specific to tool execution, includes `call_id`.
