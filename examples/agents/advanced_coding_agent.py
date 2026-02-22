import asyncio
from typing import Any

from pydantic import Field

from markov_agent import ADKConfig, BaseState, Graph
from markov_agent.containers.self_correction import CritiqueResult


class CodingState(BaseState):
    task: str = ""
    code: str = ""
    lint_output: str = ""
    test_output: str = ""
    is_lint_valid: bool = True
    is_test_valid: bool = False
    iterations: int = 0
    feedback: str = Field(default="", json_schema_extra={"behavior": "append"})


async def run_lint(state: CodingState) -> dict[str, Any]:
    """Simulate a linter."""
    # Dummy logic: if code contains 'print', it's 'invalid' for this demo
    if "print" in state.code and "import logging" not in state.code:
        return {
            "is_lint_valid": False,
            "lint_output": "Use logging instead of print.",
            "feedback": "Linter: Use logging instead of print.\n",
        }
    return {"is_lint_valid": True, "lint_output": "Clean."}


async def run_tests(state: CodingState) -> dict[str, Any]:
    """Simulate test execution."""
    state.iterations += 1
    # Dummy logic: only valid if iterations > 1
    if state.iterations < 2:
        return {
            "is_test_valid": False,
            "test_output": "AssertionError: expected 42, got 0",
            "feedback": f"Test attempt {state.iterations} failed.\n",
        }
    return {"is_test_valid": True, "test_output": "All tests passed."}


def create_coding_agent() -> Graph:
    def mock_dev(prompt):
        if "Linter" in prompt or "Feedback" in prompt:
            return "import logging\ndef get_42():\n    logging.info('returning 42')\n    return 42"
        return "def get_42():\n    print('returning 42')\n    return 42"

    def mock_reviewer(prompt):
        if "logging" in prompt:
            return '{"is_valid": true, "feedback": "Good job."}'
        return (
            '{"is_valid": false, "feedback": "Still using print or no code provided."}'
        )

    g = Graph(
        "CodingAgent",
        state_type=CodingState,
        default_adk_config=ADKConfig(
            model_name="gemini-3-flash-preview", enable_logging=False
        ),
    )

    @g.node(output_key="code", mock_responder=mock_dev)
    async def developer(state: CodingState):
        """
        You are an expert developer.
        Task: {{ task }}
        Feedback: {{ feedback }}
        Current Code: {{ code }}

        Generate the Python code to solve the task.
        """

    @g.node(output_schema=CritiqueResult, mock_responder=mock_reviewer)
    async def code_reviewer(state: CodingState):
        """
        Review the following code:
        {{ code }}

        Linter said: {{ lint_output }}

        Is the code high quality and addresses the linter concerns?
        Return is_valid=True/False and feedback string.
        """

    @g.task(name="linter")
    async def linter_task(state: CodingState):
        return await run_lint(state)

    @g.task(name="test_runner")
    async def test_task(state: CodingState):
        return await run_tests(state)

    # Use self-correction for the developer -> reviewer loop
    # We want to ensure it passes linting before moving to tests
    g.self_correction(
        primary=developer,
        critique=code_reviewer,
        name="lint_correction",
        max_retries=2,
    )
    g.entry_point = "lint_correction"

    # Add transitions
    g.add_transition("lint_correction", "test_runner")

    # If tests fail, go back to dev_loop
    g.add_transition(
        "test_runner", "lint_correction", condition=lambda s: not s.is_test_valid
    )

    return g


async def main():
    agent = create_coding_agent()
    print("Graph topology:")
    print(agent.to_mermaid())

    initial_state = CodingState(task="Create a function that returns 42.")

    final_state = await agent.run(initial_state)

    print("\nExecution History:")
    for step in final_state.history:
        print(f"Node: {step.get('node')}")

    print(f"\nFinal Code:\n{final_state.code}")
    print(f"Tests Passed: {final_state.is_test_valid}")
    print(f"Iterations: {final_state.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
