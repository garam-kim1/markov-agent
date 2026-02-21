# ADK Evaluation Spec (Python)

## 1. Evaluation Data
Two primary formats for defining test cases.

### Test Files (`*.test.json`)
Simple, single-turn cases for unit testing.
```json
[
  {
    "test_case_id": "weather_sf",
    "input": {"text": "What is the weather in SF?"},
    "expected_output": {"text": "It is sunny in San Francisco."},
    "expected_tools": ["get_weather"]
  }
]
```

### Evalsets (`*.evalset.json`)
Complex, multi-turn sessions for integration testing. Captured via `adk web` or defined manually.
```json
{
  "eval_set_id": "travel_planning",
  "eval_cases": [
    {
      "eval_id": "case_1",
      "turns": [
        {
            "user_input": "Book a flight to NYC",
            "expected_tool": "search_flights"
        }
      ]
    }
  ]
}
```

## 2. Core Metrics
ADK includes built-in metrics for trajectory and response quality.

| Metric | Type | Description |
| :--- | :--- | :--- |
| `tool_trajectory_avg_score` | Deterministic | Exact/ordered match of tool calls (0.0 - 1.0). |
| `response_match_score` | Deterministic | ROUGE-1 lexical overlap with reference. |
| `final_response_match_v2` | LLM-Judged | Semantic equivalence check by a judge model. |
| `rubric_based_quality_v1` | LLM-Judged | Custom rubric (e.g., "Is the tone professional?"). |
| `safety_v1` | Service | Vertex AI Safety check. |

## 3. Execution Methods

### CLI Batch Run
Run all evalsets in the current directory.
```bash
uv run adk eval --path ./tests/
```

### Pytest Integration
For CI/CD pipelines, wrap evaluations in `pytest`.
```python
from google.adk.eval import AgentEvaluator

def test_agent_trajectory():
    evaluator = AgentEvaluator(metrics=["tool_trajectory_avg_score"])
    result = evaluator.evaluate(
        agent=my_agent,
        test_file="tests/weather.test.json"
    )
    assert result.score > 0.9
```

### Interactive UI
Use `adk web` to record sessions as evalsets and run visual comparisons.
