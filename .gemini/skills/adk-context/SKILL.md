---
name: adk-context
description: Expert in ADK state management and service access via InvocationContext in Python. Use for multi-turn conversational memory, passing data between agents/tools, and identity tracking.
---

# ADK Context Manager

## Philosophy & Architecture
`InvocationContext` provides the "bundle" of information (session state, events, services) for a single request-response cycle (invocation).

## Context Flavors
- **`InvocationContext`**: Comprehensive internal container; used directly in `_run_async_impl`.
- **`CallbackContext`**: Shared across agent and model callbacks.
- **`ToolContext`**: Subclass of `CallbackContext` for tool callbacks; adds `function_call_id`.
- **`ReadonlyContext`**: Safe access for toolsets/services that don't modify state.

## State Management
- **Access**: `context.state` (dict-like object).
- **Modification**: `context.state['key'] = 'value'`. Changes are tracked as `state_delta` in events.
- **Prefixes**:
  - `user:*`: User-level preferences (persistent).
  - `app:*`: Application-wide settings (persistent).
  - `temp:*`: Short-lived invocation-level data.
  - (No prefix): Session-specific data.

## Identity & Tracking
- `invocation_id`: Unique identifier for the current run.
- `agent.name`: Identify the currently running agent.
- Read `references/context.md` for full attribute definitions.

## Success Criteria
- Correct state scope usage (e.g., `user:` for preferences).
- Proper access to services (e.g., `context.artifact_service`).
- Effective use of `InvocationContext` for early termination (`ctx.end_invocation = True`).
