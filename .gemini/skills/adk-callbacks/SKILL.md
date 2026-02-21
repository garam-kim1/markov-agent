---
name: adk-callbacks
description: Expert in ADK lifecycle hooks and plugins for observation and control. Use to implement guardrails, caching, and state modification in Python.
---

# ADK Interception & Lifecycle

## Philosophy
Callbacks (Local) and Plugins (Global) allow for surgical interception of agent execution.

## Logic Flow
1. **Scope Determination**:
   - Use **Plugins** for cross-cutting features (global logging, security policies).
   - Use **Callbacks** for agent-specific logic (task-dependent validation).
2. **Context Selection**:
   - Load `references/hooks.md` for hook signatures and return object types.
3. **Implementation**:
   - Implement `before_*` for validation/short-circuiting.
   - Implement `after_*` for post-processing/cleanup.

## Output Standards
- Explicit return types to signal intent (None vs. Object).
- Non-blocking async implementation.
- Correct error handling via `on_*_error` hooks.
