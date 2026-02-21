---
name: adk-plugins
description: Expert in ADK Plugins for implementing global cross-cutting features and security guardrails in Python. Use for global logging, policy enforcement, response caching, and prompt augmentation.
---

# ADK Plugin Architect

## Philosophy & Architecture
Plugins are modular code blocks executed at various stages of an agent lifecycle. While callbacks are agent-specific, plugins are **global** and registered once on the `Runner`.

## Key Capabilities
- **Precedence**: Plugins run **before** agent-level callbacks.
- **Global Scope**: Applies to all agents, tools, and LLMs managed by the `Runner`.
- **Flow Control**: Intervene in the workflow (e.g., return cached responses) or amend context (e.g., add global instructions).

## Callback Hooks
- `on_user_message_callback`: First hook to run, inspect/modify raw user input.
- `before_run_callback` / `after_run_callback`: Setup and teardown.
- `before_agent` / `after_agent`: Intercept agent-level execution.
- `on_model_error` / `on_tool_error`: Suppression of exceptions or graceful recovery.

## Prebuilt Plugins
- `LoggingPlugin`: Detailed traces.
- `ContextFilterPlugin`: Context window compression.
- `GlobalInstructionPlugin`: App-level prompts.
- `ReflectAndRetryPlugin`: Tool failure recovery.

## Success Criteria
- Valid implementation of `BasePlugin`.
- Correct registration in the `Runner` (`plugins=[MyPlugin()]`).
- Successful interception of global execution flows.
- Read `references/plugins.md` for hook definitions.
