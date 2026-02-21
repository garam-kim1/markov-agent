---
name: adk-streaming
description: Expert in ADK bidirectional (Bidi) streaming and Gemini Live API integration in Python. Use for implementing low-latency voice/video interaction and multimodal AI agents.
---

# ADK Streaming Specialist

## Philosophy & Architecture
Bidi-streaming (Live mode) adds low-latency voice and video interaction. Users can interrupt the agent, and agents can process text, audio, and video inputs in real-time.

## Key Components
- **`LiveRequestQueue`**: Upstream flow for text, audio, and video.
- **`run_live()`**: Processing events, transcriptions, and multi-agent workflows.
- **`RunConfig`**: Configure response modalities and context compression.

## Implementation Workflow
1. **Setup**: Use a FastAPI-based server for WebSocket communication.
2. **Input**: Stream multimodal data via `LiveRequestQueue`.
3. **Handling**: Process events from `run_live()` for real-time reactions.
4. **Tools**: Use "Streaming Tools" for agents to react to intermediate results (e.g., video changes).

## Best Practices
- Use `gemini-2.0-flash` for low-latency live interactions.
- Implement Voice Activity Detection (VAD) for natural turn-taking.
- Read `references/streaming.md` for part-by-dev-guide series.

## Success Criteria
- Valid implementation of real-time event loops.
- Successful handling of audio/video buffers.
- Low-latency response generation with interruption support.
