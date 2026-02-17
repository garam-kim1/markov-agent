import pytest
from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import (
    ADKConfig,
    ADKController,
    MockLlm,
    RetryPolicy,
)
from markov_agent.engine.ppu import ProbabilisticNode


class ReasoningMockLlm(MockLlm):
    """Uses base MockLlm which now handles thought parts via mock_responder."""


@pytest.mark.asyncio
async def test_adk_controller_handles_reasoning():
    config = ADKConfig(model_name="mock-reasoning")
    controller = ADKController(config, RetryPolicy())

    # We use a mock_responder that returns a dict to trigger the new MockLlm logic
    controller.agent.model = ReasoningMockLlm(
        mock_responder=lambda x: {
            "thought": "Thinking process...",
            "text": "Final answer",
        },
        model="mock-reasoning",
    )

    # Result should only be the text
    result = await controller.generate("Tell me something")
    assert result == "Final answer"

    # Reasoning should be in the state if we include it
    result, final_state = await controller.generate(
        "Tell me something", include_state=True
    )
    assert result == "Final answer"
    assert final_state["meta"]["reasoning"] == "Thinking process..."


@pytest.mark.asyncio
async def test_probabilistic_node_with_reasoning():
    node = ProbabilisticNode(name="test", prompt_template="test")
    node.controller.agent.model = ReasoningMockLlm(
        mock_responder=lambda x: {
            "thought": "Thinking process...",
            "text": "Final answer",
        },
        model="mock-reasoning",
    )

    state = BaseState()
    new_state = await node.execute(state)

    # The output in history should only be the final answer
    assert new_state.history[-1]["output"] == "Final answer"

    # The reasoning should be in the meta (synced by ProbabilisticNode)
    assert new_state.meta["reasoning"] == "Thinking process..."


class MockReasoningOutput(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_probabilistic_node_with_reasoning_and_schema():
    node = ProbabilisticNode(
        name="test", prompt_template="test", output_schema=MockReasoningOutput
    )
    # Mock responder returns JSON string for the schema
    node.controller.agent.model = ReasoningMockLlm(
        mock_responder=lambda x: {
            "thought": "Thinking process...",
            "text": '{"answer": "Final answer"}',
        },
        model="mock-reasoning",
    )

    state = BaseState()
    new_state = await node.execute(state)

    # Check if answer was parsed
    assert new_state.answer == "Final answer"

    # The reasoning should be in the meta
    assert new_state.meta["reasoning"] == "Thinking process..."

    # Also check if it's accessible via .reasoning property
    assert new_state.reasoning == "Thinking process..."
