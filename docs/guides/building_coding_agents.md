# Building Coding Agents with Markov Engine

This guide demonstrates how to build reliable **Coding Agents** using the `markov-agent` framework. Unlike simple "text-to-code" scripts, a Markov-based coding agent uses a **Probabilistic Topology** to mimic the software engineering lifecycle: Analysis, Planning, Implementation, and Review.

## The Core Pattern: Iterative Refinement

Reliable code generation requires more than a single LLM call. We use a **Cyclic Graph** to implement a "Refinement Loop."

### The Workflow

1.  **Analyzer:** Reads the code/request and identifies issues or requirements.
2.  **Planner:** Creates a step-by-step plan to address the analysis.
3.  **Coder:** Generates the code based *strictly* on the plan.
4.  **Reviewer:** Critiques the generated code (syntax, logic, style).
5.  **Router (Edge):**
    *   If the review passes (High Score) -> **Success**.
    *   If the review fails (Low Score) -> **Loop back** to Coder (or Planner) with feedback.

## 1. Define the State

The `State` is the memory of your agent. For coding, we need to track the code versions, plans, and feedback.

```python
from markov_agent.core.state import BaseState

class CodeState(BaseState):
    original_code: str
    analysis: str | None = None
    plan: str | None = None
    current_code: str | None = None
    review_feedback: str | None = None
    quality_score: int = 0
    iteration: int = 0  # To prevent infinite loops
```

## 2. Create Specialized Nodes

Use `ProbabilisticNode` and override `parse_result` to handle the specific logic of each step.

### The Coder Node (Extracting Code)

LLMs often wrap code in Markdown blocks. The `CoderNode` should extract this cleanly.

```python
import re
from markov_agent.engine.ppu import ProbabilisticNode

class CoderNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        # Extract code block logic
        code_match = re.search(r"```python(.*?)```", result, re.DOTALL)
        if code_match:
            state.current_code = code_match.group(1).strip()
        else:
            state.current_code = result  # Fallback
            
        state.iteration += 1
        return state
```

### The Reviewer Node (Structured Scoring)

The Reviewer should output a structured assessment. You can parse text or use `output_schema` for JSON mode.

```python
class ReviewerNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        # Example parsing logic
        score_match = re.search(r"Score:\s*(\d+)", result)
        state.quality_score = int(score_match.group(1)) if score_match else 0
        state.review_feedback = result
        return state
```

## 3. Construct the Topology

Wire the nodes together with `Graph` and `Edge`.

```python
from markov_agent.topology.graph import Graph
from markov_agent.topology.edge import Edge

# Define Edges
edges = [
    Edge(source="analyzer", target_func=lambda s: "planner"),
    Edge(source="planner", target_func=lambda s: "coder"),
    Edge(source="coder", target_func=lambda s: "reviewer"),
    # The Feedback Loop
    Edge(
        source="reviewer",
        target_func=lambda s: (
            "coder" if s.quality_score < 8 and s.iteration < 3 else None
        ),
    ),
]

# Create Graph
graph = Graph(
    name="code_improver",
    nodes={...}, # Your initialized nodes
    edges=edges,
    entry_point="analyzer",
    max_steps=15 # Safety limit
)
```

## 4. Best Practices for Coding Agents

*   **Temperature Control:**
    *   **Analyzer/Planner:** High temperature (0.7-0.9) for creativity and broad thinking.
    *   **Coder:** Low temperature (0.0-0.2) for strict syntax adherence.
*   **Pass@K:** Use `samples=3` for the `CoderNode` and a `ReviewerNode` to pick the best implementation if you have a way to verify them (e.g., unit tests).
*   **Mocking:** Always test your topology with a `MockLLM` before burning API credits. See `examples/code_improver_agent.py` for a full mocking example.

## See Also

*   `examples/code_improver_agent.py`: A complete, runnable implementation of this pattern.
