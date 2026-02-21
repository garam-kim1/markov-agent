---
name: adk-sessions
description: Expert in ADK session lifecycle management, multi-turn history, and persistent memory in Python. Use for implementing conversation threads, long-term state, and memory search.
---

# ADK Session Architect

## Philosophy & Architecture
ADK maintains conversational context via three pillars:
- **`Session`**: The current thread (events + state).
- **`State`**: Data within the session (`session.state`).
- **`Memory`**: Searchable cross-session knowledge.

## Core Services
1. **`SessionService`**: Manages the lifecycle of threads.
   - `InMemorySessionService`: Fast, ephemeral (for dev/test).
   - `DatabaseSessionService`: Persistent (SQL-based).
2. **`MemoryService`**: Knowledge retrieval and ingestion.
   - `search_memory(query)`: Retrieve relevant snippets from past interactions.

## State Scoping
- **Session-level**: `context.state['key']`. Default.
- **User-level**: `user:pref`. Persistent across sessions for the same user.
- **App-level**: `app:setting`. Global for all users of the application.

## Best Practices
- Use `InMemoryRunner` for local prototyping.
- Leverage the Jinja2 template `{var}` syntax in agent instructions to inject state automatically.
- Read `references/sessions.md` for service configuration patterns.

## Success Criteria
- Correct session creation and retrieval logic via `Runner`.
- Effective use of `MemoryService` for grounding.
- Proper scoping of state to prevent cross-talk.
