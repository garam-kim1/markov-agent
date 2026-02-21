# ADK Context & State Spec (Python)

## State Scoping (`context.state`)
- **Session (Default)**: `state['key']`.
- **User (Cross-Session)**: `state['user:pref']`.
- **App (Global)**: `state['app:config']`.
- **Temporary (Invocation)**: `state['temp:val']`.

## Context Properties
- `invocation_id`: Tracking ID.
- `agent`: Current agent instance.
- `session.id`: Conversation ID.
- `services`: Access to `artifact_service`, `memory_service`, `session_service`.

## Control Signals
- `ctx.end_invocation = True`: Gracefully stop the current run.

## Access Patterns
```python
# In a tool
def my_tool(tool_context):
    user_id = tool_context.state.get('user:id')
    tool_context.state['temp:last_action'] = 'fetch'
```

## Template Injection
Use `{var}` in Agent instructions. ADK replaces with `state[var]`.
- `{var?}`: Optional (no error if missing).
- `{artifact.name}`: Inject text content of artifact.
