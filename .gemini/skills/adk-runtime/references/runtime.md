# Runtime & Configuration (Python)

## 1. The Runner
The `Runner` orchestrates the conversation. It connects the `Agent`, `Session`, and `User`.

### Initialization
```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
runner = Runner(
    agent=my_agent,
    session_service=session_service,
    app_name="my_app"
)
```

### Execution Loop (`run_async`)
The runner yields `Event` objects. You must iterate through them.

```python
from google.adk.models import Content, Part

user_msg = Content(role="user", parts=[Part(text="Hello!")])

async for event in runner.run_async(
    user_id="user_123",
    session_id="session_abc",
    new_message=user_msg
):
    # Events can be: "chunk" (streaming), "tool_call", "log", "final_response"
    if event.is_final_response():
        print(f"Agent: {event.content.parts[0].text}")
    elif event.type == "tool_call":
        print(f"Tool used: {event.data['name']}")
```

## 2. RunConfig
Control runtime behavior per-request.

```python
from google.adk.types import RunConfig

config = RunConfig(
    streaming_mode="word_level",  # or 'turn_level'
    context_window_strategy="sliding_window",
    max_turns=10
)

# Pass config to run_async
runner.run_async(..., run_config=config)
```

## 3. Session Management
Sessions store state (`ctx.session.state`) and history (`ctx.session.history`).
- **`InMemorySessionService`**: Good for local dev/testing.
- **Custom Service**: Implement `SessionService` interface for Redis/SQL storage.

## 4. Hosting (FastAPI)
ADK integrates easily with FastAPI for serving agents as APIs.

```python
from fastapi import FastAPI
from google.adk.integrations.fastapi import create_agent_route

app = FastAPI()

# Creates POST /agent/chat
app.include_router(
    create_agent_route(runner, path="/agent/chat")
)
```
