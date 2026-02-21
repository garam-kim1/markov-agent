---
name: adk-evaluate
description: Expert in ADK agent evaluation, trajectory analysis, and metric-based testing in Python. Use for implementing unit/integration tests for AI agents, measuring tool accuracy, and grounding checks.
---

# ADK Evaluation Specialist

## Philosophy & Architecture
Agent evaluation requires qualitative assessment of both the final output and the **trajectory** (sequence of steps/tool calls).

## Evaluation Methods
1. **Test Files (`*.test.json`)**:
   - Single simple session for rapid development.
   - Contains: `User Content`, `Expected Trajectory`, `Final Response`.
2. **Evalsets (`*.evalset.json`)**:
   - Complex, multi-turn sessions for integration testing.
   - Ideal for regression testing in CI/CD pipelines.

## Core Metrics
- `tool_trajectory_avg_score`: Exact match of expected tool calls.
- `response_match_score`: ROUGE-1 similarity.
- `final_response_match_v2`: LLM-judged semantic match.
- `hallucinations_v1`: Groundedness against context.
- `safety_v1`: Harmlessness assessment.
- Read `references/evaluate.md` for scoring logic.

## Execution
- **`adk web`**: Interactive UI for recording and running evals.
- **`adk eval`**: CLI command for automated batch testing.
- **`pytest`**: Integration with `AgentEvaluator.evaluate(...)`.

## Success Criteria
- Valid JSON test files adhering to the ADK Pydantic schema.
- Comprehensive coverage of critical "happy path" trajectories.
- Successful integration with automated CI triggers.
