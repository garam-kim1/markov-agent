---
name: adk-runtime
description: Expert in ADK agent runtime configuration, execution modes, and server operations in Python. Use for configuring the Dev UI, CLI runs, and RESTful API servers.
---

# ADK Runtime Architect (Python Edition)

## Philosophy
The Runtime is the engine that drives agent execution. It manages the `Session`, `Event` loop, and `RunConfig`.

## Logic Flow
1. **Select Mode**:
   - **CLI**: `Runner.run_async()` loop.
   - **Service**: `FastAPIServer` or `adk api_server`.
   - **Interactive**: `run_live()` (Websockets/Bidi).
2. **Configure**: Use `RunConfig` to control behavior.
3. **Execution**: Implement the loop to handle `Event` objects.

## References
- Read `references/runtime.md` for deep details on `Runner`, `RunConfig`, and `Event`.

## Standards
- Always use `async` runners.
- Handle `event.is_final_response()` to detect completion.
- Use `InMemorySessionService` for dev, implementation-specific for prod.
