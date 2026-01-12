import asyncio

import pytest
from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.sampler import execute_parallel_sampling


# --- Test Data ---
class StateForTest(BaseState):
    query: str
    response: str = ""


# --- Tests ---


@pytest.mark.asyncio
async def test_adk_controller_mock():
    config = ADKConfig(model_name="mock-model")
    retry = RetryPolicy()

    def mock_resp(p):
        return "Mock response for test"

    controller = ADKController(config, retry, mock_responder=mock_resp)

    response = await controller.generate("Hello")
    assert "Mock response" in response


@pytest.mark.asyncio
async def test_parallel_sampling():
    count = 0

    async def task():
        nonlocal count
        count += 1
        await asyncio.sleep(0.01)
        return count

    # Run 5 times
    results = await execute_parallel_sampling(task, k=5, selector_func=lambda x: x)
    assert len(results) == 5
    assert count == 5


@pytest.mark.asyncio
async def test_probabilistic_node():
    config = ADKConfig(model_name="mock-model")

    def mock_resp(p):
        return "Mock response for node test"

    node = ProbabilisticNode(
        name="test_node",
        adk_config=config,
        prompt_template="User said: {query}",
        samples=2,
        mock_responder=mock_resp,
    )

    # Custom parser for test
    def custom_parser(state, result):
        return state.update(response=result)

    node.state_updater = custom_parser

    state = StateForTest(query="Hello World")
    new_state = await node.execute(state)

    assert "Mock response" in new_state.response


@pytest.mark.asyncio
async def test_structured_output():
    class OutputModel(BaseModel):
        answer: str
        confidence: float

    # Mock responder that returns JSON
    def json_mock(prompt):
        return '{"answer": "yes", "confidence": 0.95}'

    config = ADKConfig(model_name="mock-model")
    node = ProbabilisticNode(
        name="structure_node",
        adk_config=config,
        prompt_template="Answer {query}",
        output_schema=OutputModel,
        mock_responder=json_mock,
    )

    state = StateForTest(query="Is this real?")
    new_state = await node.execute(state)

    # Check that output in history is a dict (parsed from model)
    last_step = new_state.history[-1]
    assert last_step["node"] == "structure_node"
    assert last_step["output"] == {"answer": "yes", "confidence": 0.95}
