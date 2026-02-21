---
name: adk-agents
description: Expert in ADK agent architectures. Use to design and implement LlmAgents (reasoning), WorkflowAgents (deterministic), or CustomAgents (complex branching) in Python.
---

# ADK Agent Architect

## Philosophy
Agents are modular execution units. Use the specialized references below to implement the specific architecture required for the task.

## Logic Flow
1. **Analyze Complexity**:
   - For **FSM-based topology** (Markovian routing) -> Use the `markov-topology` skill.
   - For dynamic reasoning/tool use -> `LlmAgent`.
   - For fixed order/loop/parallel execution -> `WorkflowAgent`.
   - For non-standard branching -> `CustomAgent`.
2. **Context Loading**:
   - If working on **LLM-driven agents**, read `references/llm-agents.md`.
   - If working on **Workflows (Sequential/Parallel/Loop)**, read `references/workflow-agents.md`.
   - If working on **Inheritance/Custom Logic**, read `references/custom-agents.md`.
3. **Implementation**: Generate Python 3.12+ code using `google-adk`.

## Output Standards
- Explicit `InvocationContext` handling.
- Use of Jinja2 `{var}` templates in instructions.
- Strict async implementation using `AsyncGenerator[Event, None]`.
