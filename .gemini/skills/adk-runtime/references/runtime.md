# ADK Runtime & Config Spec (Python)

## Running Modes
1. **Dev UI (`adk web <folder>`)**: Browser-based interaction and debugging.
2. **CLI (`adk run <module>`)**: Terminal-based interaction.
3. **API Server (`adk api_server`)**: RESTful API for integration.

## Core API (`Runner`)
- **`run_async(...)`**: Main execution loop.
- **`run_live(...)`**: Bidi-streaming mode.
- **`resume(...)`**: Restore context from `session_id`.

## Configuration (`RunConfig`)
- `response_modalities`: Text, Audio, Image.
- `streaming_mode`: Real-time vs. Turn-based.
- `context_window`: Compression/quota management.

## Implementation Pattern
```python
# Launching the API Server
# adk api_server samples_for_testing/hello_world
```
```python
# Programmatic run
async for event in runner.run_async(user_id="U1", session_id="S1", new_message=msg):
    if event.is_final_response():
        print(event.content)
```
