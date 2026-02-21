# ADK Safety & Security Spec (Python)

## 1. Identity & Authorization
Control *who* executes actions.

### Tool Context Auth
Use `ToolContext` to validate the invoking user against the tool's required permissions.
```python
def delete_user(user_id: str, tool_context: ToolContext):
    # Check if the session user is an admin
    if tool_context.session.state.get("user_role") != "admin":
        return {"error": "Unauthorized: Admin role required"}
    # ... proceed
```

### Service Account Delegation
When running in Google Cloud, agents use the environment's Service Account. Ensure `Least Privilege` via IAM roles.

## 2. Gemini Safety Filters
Configure the model to block harmful content at the generation layer.

```python
from google.genai.types import HarmCategory, HarmBlockThreshold

safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }
]

agent = LlmAgent(..., safety_settings=safety_settings)
```

## 3. Input/Output Guardrails (Pydantic)
Prevent injection attacks and ensure valid data structures.

- **Input**: Use `pydantic` models for tool arguments. ADK automatically validates types.
- **Output**: Use `output_schema` to force structured JSON, reducing hallucination risk.

## 4. Sandboxed Code Execution
**NEVER** execute LLM-generated code locally in production. Use `AgentEngineSandboxCodeExecutor`.

```python
from google.adk.tools import AgentEngineSandboxCodeExecutor

code_tool = AgentEngineSandboxCodeExecutor()
agent = LlmAgent(..., tools=[code_tool])
```

## 5. Callback Validation
Intercept execution to enforce policies dynamically.

```python
def validate_inputs(callback_context, tool, args, tool_context):
    if "DROP TABLE" in str(args):
        # Block the tool execution
        return {"error": "SQL Injection Detected"}
    return None # Allow execution

agent = LlmAgent(..., before_tool_callback=validate_inputs)
```

## 6. Output Escaping
**Always** HTML-escape agent output before rendering in a UI to prevent XSS.
```python
import html
safe_output = html.escape(agent_response_text)
```
