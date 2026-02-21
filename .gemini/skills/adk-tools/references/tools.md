# Tools & Toolsets (Python)

## 1. Function Tools (The Standard Way)
The easiest way to create a tool is a standalone Python function.

### Requirements
- **Type Hints**: ADK uses `pydantic` to inspect signatures and generate the JSON schema for the LLM.
- **Docstring**: The LLM reads this to understand *how* to use the tool.
- **Return Type**: `dict` is best practice for structured feedback.

### Example
```python
def search_product(name: str, max_results: int = 5) -> dict:
    """
    Searches the product catalog by name.

    Args:
        name: The partial name of the product.
        max_results: Max number of items to return.
    """
    # Logic here...
    return {"results": ["item1", "item2"]}
```

## 2. Tool Context (`ToolContext`)
To access the agent's state or control the flow *from within the tool*, add `tool_context: ToolContext` as an argument. The framework injects this automatically; the LLM does *not* see it.

### Capabilities
- **`tool_context.call_id`**: Unique ID of the tool call.
- **`tool_context.session.state`**: Read/write global session state.
- **`tool_context.actions`**: Control agent behavior.

### Flow Control Actions
- `actions.transfer_to_agent = "agent_name"`: Hand off execution.
- `actions.skip_summarization = True`: The tool's output becomes the final response (no LLM rewriting).
- `actions.escalate = True`: Signal a parent agent or stop processing.

### Example
```python
from google.adk.types import ToolContext

def update_user_preference(key: str, value: str, tool_context: ToolContext) -> dict:
    """Updates a user preference setting."""
    
    # 1. Persist to session state
    tool_context.session.state[f"pref_{key}"] = value

    # 2. Skip summarization (return strict confirmation)
    tool_context.actions.skip_summarization = True

    return {"status": "updated", "key": key}
```

## 3. BaseTool (Class-Based Tools)
Inherit from `BaseTool` for complex tools requiring setup (e.g., database connections).

```python
from google.adk.tools import BaseTool

class DatabaseTool(BaseTool):
    def __init__(self, db_connection):
        super().__init__(name="query_db", description="Executes SQL queries.")
        self.conn = db_connection

    def execute(self, query: str) -> dict:
        cursor = self.conn.cursor()
        cursor.execute(query)
        return {"rows": cursor.fetchall()}
```

## 4. BaseToolset (Dynamic Groups)
Group related tools or load them dynamically.

```python
from google.adk.tools import BaseToolset

class AdminToolset(BaseToolset):
    def get_tools(self, ctx) -> list:
        # Only return admin tools if user is admin
        if ctx.session.state.get("role") == "admin":
            return [delete_user, reset_system]
        return []
```
