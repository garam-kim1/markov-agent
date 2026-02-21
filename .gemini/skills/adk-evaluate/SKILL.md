---
name: adk-evaluate
description: Expert in ADK agent evaluation, trajectory analysis, and metric-based testing in Python. Use for implementing unit/integration tests for AI agents, measuring tool accuracy, and grounding checks.
---

# ADK Evaluation Specialist (Python Edition)

## Philosophy & Architecture
Agent evaluation requires qualitative assessment of both the final output and the **trajectory** (sequence of steps/tool calls).

## Evaluation Methods
1. **Test Files (`*.test.json`)**:
   - Single-turn sessions for rapid development.
   - Ideal for unit testing via `pytest`.
2. **Evalsets (`*.evalset.json`)**:
   - Complex, multi-turn sessions for integration testing.
   - Ideal for regression testing in CI/CD pipelines via `adk eval`.

## Core Metrics
- `tool_trajectory_avg_score`: Exact match of expected tool calls (0.0 - 1.0).
- `response_match_score`: ROUGE-1 similarity.
- `final_response_match_v2`: LLM-judged semantic match.
- `hallucinations_v1`: Groundedness against context.
- `safety_v1`: Harmlessness assessment.
- Read `references/evaluate.md` for full metric details and data schemas.

## Execution
- **`adk web`**: Interactive UI for recording sessions and visual comparisons.
- **`adk eval`**: CLI command for automated batch testing.
- **`pytest`**: Integration with `AgentEvaluator.evaluate(...)` for CI/CD.

## Success Criteria
- Valid JSON test files adhering to the ADK Pydantic schema.
- Comprehensive coverage of critical "happy path" trajectories.
- Successful integration with automated CI triggers.
