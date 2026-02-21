# ADK Safety & Security Spec (Python)

## Safety Layers
1. **Identity & Auth**:
   - `Agent-Auth`: Tool uses a Service Account.
   - `User-Auth`: Tool uses user OAuth tokens.
2. **In-tool Guardrails**:
   - Design tools defensively.
   - Access policies via `tool_context.state`.
3. **Gemini Safety Filters**:
   - Configurable thresholds for hate speech, harassment, sexually explicit, and dangerous content.
4. **Sandboxed Code Execution**:
   - `Vertex Gemini Enterprise API` or `Vertex Code Interpreter Extension`.
5. **Plugins**:
   - `Gemini as a Judge`: LLM-based input/output screening.
   - `Model Armor`: Harmful content detection.
   - `PII Redaction`: Filter sensitive data before tool calls.

## Best Practices
- **UI Safety**: Always escape model-generated HTML/JS.
- **VPC-SC**: Execute agents inside secure network perimeters.
- **Evaluation**: Use `adk-evaluate` for groundedness and safety metrics.

## Implementation Pattern
```python
# Before Tool Callback for validation
def validate_tool_params(callback_context, tool, args, tool_context):
    if args.get("user_id") != callback_context.state.get("session_user_id"):
        return {"error": "Unauthorized Access"}
    return None
```
