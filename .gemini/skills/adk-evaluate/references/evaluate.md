# ADK Evaluation Spec (Python)

## Evaluation Types
1. **Test Files (`*.test.json`)**: Unit tests. Single interaction (turn).
2. **Evalsets (`*.evalset.json`)**: Integration tests. Complex multi-turn sessions.

## Core Metrics (Built-in)
- `tool_trajectory_avg_score`: Exact match of tool sequence.
- `response_match_score`: ROUGE-1 similarity (0.8 default).
- `final_response_match_v2`: LLM-judged semantic equivalence.
- `hallucinations_v1`: Groundedness against tool context.
- `safety_v1`: Policy compliance check.

## Data Schema
```json
{
  "eval_set_id": "id",
  "eval_cases": [
    {
      "eval_id": "case_1",
      "conversation": [
        {
          "user_content": {"parts": [{"text": "query"}]},
          "final_response": {"parts": [{"text": "expected"}]},
          "intermediate_data": {"tool_uses": [{"name": "tool", "args": {}}]}
        }
      ]
    }
  ]
}
```

## Tools
- `adk web`: Interactive capture of sessions to evalsets.
- `adk eval`: CLI batch execution.
- `pytest`: Programmatic integration via `AgentEvaluator.evaluate()`.
