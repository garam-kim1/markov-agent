# ADK Bidi-Streaming & Live Spec (Python)

## Technology
Integrates **Gemini Live API** for low-latency voice/video interaction.

## Core API
- **`LiveRequestQueue`**: Send multimodal input (text, audio, image/video).
- **`run_live()`**: Handle real-time event stream.
- **`Streaming Tools`**: Agents react to intermediate tool results (e.g., video stream changes).

## Implementation Pattern (FastAPI)
1. **Initialize**: `LiveRequestQueue` for user input.
2. **Execute**: `runner.run_live(request_queue, run_config)`.
3. **Handle**: Iterate over `AsyncGenerator[Event, None]`.

## Configuration (`RunConfig`)
- `voice_name`: Select agent voice (e.g., "Puck", "Charon").
- `response_modalities`: `[MODALITY_AUDIO]`.
- `speech_config`: Thresholds for voice activity detection (VAD).

## Use Cases
- Natural human-like voice conversations.
- Interruption handling (User speaks, Agent pauses).
- Multimodal reasoning (What am I looking at right now?).
