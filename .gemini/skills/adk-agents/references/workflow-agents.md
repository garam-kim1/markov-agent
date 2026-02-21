# Workflow Agents (Python)

Workflow agents orchestrate execution flow deterministically. They do *not* use an LLM to decide the next step; the logic is hard-coded by the agent type.

## 1. `SequentialAgent`
Executes a list of agents in strict order. Passes state between them via `session.state`.

```python
from google.adk.agents import SequentialAgent

flow = SequentialAgent(
    name="research_flow",
    agents=[research_agent, writer_agent, editor_agent]
)
```

## 2. `ParallelAgent`
Executes multiple agents concurrently. Useful for aggregating data from multiple sources.
- **`max_workers`**: Control concurrency level.
- **Merge Strategy**: Results are stored in `session.state` based on each sub-agent's `output_key`.

```python
from google.adk.agents import ParallelAgent

parallel_research = ParallelAgent(
    name="parallel_research",
    agents=[web_searcher, database_query, internal_docs],
    max_workers=3
)
```

## 3. `LoopAgent`
Repeats a sub-agent (or a sequence) until a condition is met.
- **`loop_condition`**: A function `(ctx) -> bool`. Returns `True` to continue looping.
- **`max_iterations`**: Safety limit.

```python
def check_quality(ctx):
    # Continue looping if quality score is low
    score = ctx.session.state.get("quality_score", 0)
    return score < 0.8

optimizer = LoopAgent(
    name="optimizer_loop",
    agent=refinement_agent,  # The agent to repeat
    condition=check_quality,
    max_iterations=5
)
```

## 4. `SwitchAgent` (Routing)
Routes execution to *one* of several agents based on a condition function.
- **`router`**: A function `(ctx) -> str`. Returns the `name` of the next agent.

```python
def route_request(ctx):
    topic = ctx.session.state.get("topic")
    if topic == "code":
        return "coding_agent"
    elif topic == "search":
        return "search_agent"
    return "chat_agent"

router = SwitchAgent(
    name="main_router",
    router=route_request,
    agents=[coding_agent, search_agent, chat_agent]
)
```
