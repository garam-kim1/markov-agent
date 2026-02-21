---
name: adk-artifacts
description: Expert in ADK binary data management and versioning in Python. Use for file handling, GCS persistence, and non-textual interaction.
---

# ADK Artifact Manager

## Philosophy
Artifacts handle file-like binary data (PDFs, images, audio) that don't fit in session state.

## Logic Flow
1. **Analyze Input**:
   - For saving binary results -> `save_artifact`.
   - For reading existing data -> `load_artifact`.
   - For directory listing -> `list_artifacts`.
2. **Context Selection**:
   - Load `references/artifacts.md` for API signatures and `types.Part` schema.
3. **Implementation**:
   - Use `types.Part.from_bytes` for raw data.
   - Use `InMemoryArtifactService` for testing.
   - Use `GcsArtifactService` for production persistence.

## Output Standards
- Explicit MIME type assignment (e.g., `"application/pdf"`).
- Error handling for missing artifact services in `Runner`.
- Valid use of versioning (latest by default).
