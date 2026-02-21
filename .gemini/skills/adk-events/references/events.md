# ADK Events Spec (Python)

## Event Schema (`google.adk.events.Event`)
Inherits from `LlmResponse`.
- `author` (str): `'user'` or agent name.
- `invocation_id` (str): Unique run ID.
- `id` (str): Unique event ID.
- `timestamp` (float): Epoch time.
- `content` (Optional[`types.Content`]): Payload (text, parts).
- `actions` (`EventActions`): State/Artifact deltas + Control Signals.
- `partial` (bool): Streaming chunk flag.
- `turn_complete` (bool): Final event for current turn.

## EventActions Schema
- `state_delta` (dict): New/modified session state.
- `artifact_delta` (dict): Saved artifact names and versions (`{filename: version}`).
- `transfer_to_agent` (str): Name of agent to hand off control.
- `escalate` (bool): Loop termination/parent hand-off signal.
- `skip_summarization` (bool): Raw tool output bypass.

## Identification
- **Text**: `event.content.parts[0].text`.
- **Tool Request**: `event.get_function_calls()`.
- **Tool Response**: `event.get_function_responses()`.
- **Final Result**: `event.is_final_response()`.

## Event Flow
`User -> Runner -> SessionService (Delta Merger) -> Application`.
History is preserved in `session.events`.
