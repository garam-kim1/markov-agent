# ADK Session & Memory Spec (Python)

## Session Pillar
Represents a single conversation thread.
- **Service**: `SessionService` (InMemory, Database, VertexAi).
- **Lifecycle**: Create, Retrieve, Update (Events/State), Delete.
- **State**: Persistent storage for key-value pairs.

## Memory Pillar
Searchable cross-session knowledge archive.
- **Service**: `MemoryService`.
- **API**: `tool_context.search_memory(query)`.
- **Grounding**: Used to recall past preferences or external docs.

## Runner Integration
```python
runner = Runner(
    agent=root_agent,
    session_service=InMemorySessionService(),
    artifact_service=InMemoryArtifactService()
)
session = await runner.session_service.create_session(user_id="123")
```

## Resilience
`SessionService.append_event(event)` persists history and applies deltas.
- **Resumption**: Re-run with `session_id` to restore context.
