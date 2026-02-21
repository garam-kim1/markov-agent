# LLM Agent (Python)

The `LlmAgent` is the cognitive core of ADK, leveraging a Large Language Model for reasoning, decision-making, and tool orchestration.

## Key Attributes (`LlmAgent`)

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | Unique identifier (snake_case). Required. |
| `model` | `str` | Model identifier (e.g., `gemini-2.0-flash-exp`). |
| `instruction` | `str` | System prompt. Supports Jinja2 templates (e.g., `{user_name}`). |
| `tools` | `list` | List of functions or `BaseTool` instances. |
| `output_key` | `str` | Key to store the final result in `session.state`. |
| `output_schema` | `type[BaseModel]` | Pydantic model for structured output enforcement. |

## Implementation Patterns

### 1. Basic Reasoning Agent
```python
from google.adk import LlmAgent

agent = LlmAgent(
    name="reasoning_agent",
    model="gemini-2.0-flash-exp",
    instruction="You are a helpful assistant. Answer broadly.",
)
```

### 2. Structured Output Agent (Pydantic)
Use `output_schema` to force the model to return a valid JSON object matching a Pydantic model. The result is automatically parsed and stored in `session.state`.

```python
from pydantic import BaseModel, Field
from google.adk import LlmAgent

class UserProfile(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age")
    interests: list[str] = Field(description="List of hobbies")

profiler = LlmAgent(
    name="profiler",
    model="gemini-2.0-flash-exp",
    instruction="Extract user details from the conversation.",
    output_schema=UserProfile,
    output_key="user_profile"  # Result stored in session.state["user_profile"]
)
```

### 3. Tool-Use Agent
Tools are Python functions. Type hints and docstrings are **mandatory** as they form the tool definition for the model.

```python
def get_weather(city: str) -> dict:
    """Returns weather for a city."""
    return {"temp": 72, "city": city}

agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash-exp",
    instruction="You are a weather bot. Use the get_weather tool.",
    tools=[get_weather]
)
```

## Advanced Configuration

### `GenerateContentConfig`
Control the generation parameters (temperature, tokens, safety).

```python
from google.genai.types import GenerateContentConfig, HarmCategory, HarmBlockThreshold

config = GenerateContentConfig(
    temperature=0.0,  # Deterministic
    max_output_tokens=1024,
    safety_settings=[
        {
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)

agent = LlmAgent(..., generate_content_config=config)
```

### `ThinkingConfig` (Planner)
Enable "Chain of Thought" or internal reasoning before the final response.

```python
from google.adk.planners import BuiltInPlanner
from google.genai.types import ThinkingConfig

planner = BuiltInPlanner(
    thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=1024)
)

agent = LlmAgent(..., planner=planner)
```

## Context Management
- **`include_contents`**:
    - `'default'`: Sends full history.
    - `'none'`: Sends only the current turn (stateless/one-shot).

```python
stateless_agent = LlmAgent(..., include_contents='none')
```
