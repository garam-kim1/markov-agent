# ADK Artifacts Spec (Python)

## Service Types (`BaseArtifactService`)
- `InMemoryArtifactService`: Local testing (lost on restart).
- `GcsArtifactService`: Google Cloud Storage (persistent).

## Core API (via `CallbackContext` / `ToolContext`)
- `save_artifact(filename: str, artifact: types.Part) -> int`: Save binary data. Returns version.
- `load_artifact(filename: str, version: Optional[int] = None) -> types.Part`: Retrieve data.
- `list_artifacts() -> list[str]`: List available keys.

## Binary Representation (`types.Part`)
```python
artifact = types.Part(
    inline_data=types.Blob(
        mime_type="image/png",
        data=image_bytes
    )
)
```

## Implementation Pattern
```python
# Save result from a tool
async def save_doc(tool_context, doc_bytes):
    part = types.Part.from_bytes(data=doc_bytes, mime_type="application/pdf")
    await tool_context.save_artifact("report.pdf", part)
```

## Persistence Logic
`SessionService` merges `artifact_delta` from events. Service must be configured in `Runner`.
- `user:` prefix: Persistent across user sessions (requires database backend).
- `filename`: Session-specific (default).
