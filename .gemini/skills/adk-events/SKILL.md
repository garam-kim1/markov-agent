---
name: adk-events
description: Expert in the ADK Event system, including event identification, payload extraction, and action detection in Python. Use for implementing observers, custom logging, and control flow signals.
---

# ADK Event Specialist

## Philosophy & Architecture
Events are immutable records representing occurrences in an agent's lifecycle. Everything—user input, LLM output, tool results, state changes—is communicated as an `Event`.

## Event Structure
A `google.adk.events.Event` contains:
- `author`: The origin (e.g., `'user'` or agent name).
- `content`: `types.Content` object (text, parts).
- `actions`: `EventActions` payload for side-effects (state/artifact deltas).
- `partial`: Boolean indicating streaming chunks.
- Read `references/events.md` for full field definitions.

## Key Identification Patterns
1. **Text Message**: `event.content.parts[0].text` exists.
2. **Tool Request**: `event.get_function_calls()` returns a non-empty list.
3. **Tool Result**: `event.get_function_responses()` returns a non-empty list.
4. **Final Response**: Use `event.is_final_response()` to determine if the event should be displayed to the end-user.

## Detecting Side Effects
Check `event.actions`:
- **State Changes**: `event.actions.state_delta` (dict).
- **Artifact Saves**: `event.actions.artifact_delta` (dict).
- **Control Signals**: `transfer_to_agent` (str) or `escalate` (bool).

## Success Criteria
- Accurate filtering of events (e.g., distinguishing between streaming chunks and final results).
- Proper handling of `function_call_id` for correlating requests and responses.
- Idiomatic extraction of content using `event.content.parts`.
