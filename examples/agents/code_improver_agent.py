import asyncio
import re

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

# --- 1. Define State ---


class CodeState(BaseState):
    original_code: str
    analysis: str | None = None
    plan: str | None = None
    current_code: str | None = None
    review_feedback: str | None = None
    quality_score: int = 0
    iteration: int = 0


# --- 2. Define Nodes ---


class AnalyzerNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        state.analysis = result
        state.record_step({"node": self.name, "output": "Analysis complete"})
        return state


class PlannerNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        state.plan = result
        state.record_step({"node": self.name, "output": "Plan created"})
        return state


class CoderNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        # Extract code block if present
        code_match = re.search(r"```python(.*?)```", result, re.DOTALL)
        if code_match:
            state.current_code = code_match.group(1).strip()
        else:
            state.current_code = result  # Fallback
        state.iteration += 1
        state.record_step({"node": self.name, "output": "Code generated"})
        return state


class ReviewerNode(ProbabilisticNode[CodeState]):
    def parse_result(self, state: CodeState, result: str) -> CodeState:
        # Expecting format: "Score: 8\nFeedback: ..."
        score_match = re.search(r"Score:\s*(\d+)", result)
        if score_match:
            state.quality_score = int(score_match.group(1))
        else:
            state.quality_score = 0  # Default to low score if parsing fails

        state.review_feedback = result
        state.record_step({"node": self.name, "score": state.quality_score})
        return state


# --- 3. Mock LLM Logic ---


class MockLLM:
    def __init__(self):
        self.attempts = 0

    def __call__(self, prompt: str) -> str:
        """Simulates different agents based on the prompt."""
        if "Analyze the following code" in prompt:
            return "The code has a bug in the loop condition and lacks type hints."

        if "Create a plan" in prompt:
            return "1. Fix the loop.\n2. Add type hints.\n3. Add docstrings."

        if "Generate Python code" in prompt:
            self.attempts += 1
            if self.attempts == 1:
                # First attempt: Generate slightly wrong code (missing type hints)
                return """Here is the code:
```python
def factorial(n):
    if n == 0: return 1
    return n * factorial(n-1)
```
"""
            # Second attempt: Fix it
            return """Here is the fixed code:
```python
def factorial(n: int) -> int:
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```
"""

        if "Review the following code" in prompt:
            # Check if code in prompt has type hints
            if "def factorial(n):" in prompt:
                return "Score: 5\nFeedback: Missing type hints. Please add them."
            return "Score: 9\nFeedback: Great job, strict typing is present."

        return "I don't know how to respond to that."


# --- 4. Topology ---


async def main():
    # Configuration
    config = ADKConfig(model_name="gemini-1.5-pro")

    # Mock
    mock_llm = MockLLM()

    # Nodes
    analyzer = AnalyzerNode(
        name="analyzer",
        adk_config=config,
        prompt_template="Analyze the following code for bugs:\n{{ original_code }}",
        mock_responder=mock_llm,
        state_type=CodeState,
    )

    planner = PlannerNode(
        name="planner",
        adk_config=config,
        prompt_template="Create a plan to fix these issues:\n{{ analysis }}",
        mock_responder=mock_llm,
        state_type=CodeState,
    )

    coder = CoderNode(
        name="coder",
        adk_config=config,
        prompt_template=(
            "Generate Python code based on plan:\n{{ plan }}\n\nFeedback: {{ review_feedback }}"
        ),
        mock_responder=mock_llm,
        state_type=CodeState,
    )

    reviewer = ReviewerNode(
        name="reviewer",
        adk_config=config,
        prompt_template="Review the following code:\n{{ current_code }}",
        mock_responder=mock_llm,
        state_type=CodeState,
    )

    # Edges
    # analyzer -> planner -> coder -> reviewer
    # reviewer -> coder (if score < 8)
    # reviewer -> END (if score >= 8)

    edges = [
        Edge(source="analyzer", target_func=lambda s: "planner"),
        Edge(source="planner", target_func=lambda s: "coder"),
        Edge(source="coder", target_func=lambda s: "reviewer"),
        Edge(
            source="reviewer",
            target_func=lambda s: (
                "coder" if s.quality_score < 8 and s.iteration < 3 else None
            ),
        ),
    ]

    graph = Graph(
        name="code_improver_graph",
        nodes={n.name: n for n in [analyzer, planner, coder, reviewer]},
        edges=edges,
        entry_point="analyzer",
        max_steps=10,
        state_type=CodeState,
    )

    # Initial State
    initial_code = """
def fact(n):
    res = 1
    i = 1
    while i < n: # Bug: should be <=
        res = res * i
        i = i + 1
    return res
"""
    state = CodeState(original_code=initial_code)

    # Execution
    final_state = await graph.run(state)

    print("\n--- Final Result ---")
    print(f"Final Code:\n{final_state.current_code}")
    print(f"Final Score: {final_state.quality_score}")
    print(f"History Steps: {len(final_state.history)}")


if __name__ == "__main__":
    asyncio.run(main())
