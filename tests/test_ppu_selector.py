
import pytest
from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode


class MockOutput(BaseModel):
    value: int


class MockState(BaseState):
    result: MockOutput | None = None


@pytest.mark.asyncio
async def test_ppu_selector_logic():
    # 1. Setup
    # We want to simulate k=3 samples: [1, 2, 2]
    # The selector should pick 2 (majority vote)

    mock_responses = [MockOutput(value=1), MockOutput(value=2), MockOutput(value=2)]

    # Mock the ADKController.generate to return these in order
    # Since ProbabilisticNode creates its own controller, we need to intercept or mock the behavior.
    # Actually ProbabilisticNode takes `mock_responder`.

    # The mock_responder in ADKController usually mimics the genai.Model.generate_content
    # But here we are deep in the stack.
    # Let's mock the `controller.generate` method directly after initialization?
    # Or better, use a custom mock_responder that yields values.

    response_iter = iter(mock_responses)

    def mock_generate(*args, **kwargs):
        return next(response_iter).model_dump_json()

    # We can't easily pass a simple async function as mock_responder because ADKController expects an object with generate_content?
    # No, ADKController takes mock_responder and uses it.
    # If mock_responder is set, ADKController.generate calls `self.mock_responder(prompt)`.
    # So we can pass the async function.

    selector_called = False

    def my_selector(results):
        nonlocal selector_called
        selector_called = True
        # Majority vote
        vals = [r.value for r in results]
        most_common = max(set(vals), key=vals.count)
        return MockOutput(value=most_common)

    node = ProbabilisticNode(
        name="test_node",
        adk_config=ADKConfig(model_name="mock"),
        prompt_template="test",
        output_schema=MockOutput,
        samples=3,
        selector=my_selector,
        state_type=MockState,
        state_updater=lambda s, r: s.model_copy(update={"result": r}),
        mock_responder=mock_generate,
    )

    # 2. Execute
    initial_state = MockState()
    # We need to use `execute` or run properly.
    # execute() wraps execute_parallel_sampling.

    new_state = await node.execute(initial_state)

    # 3. Verify
    assert selector_called
    assert new_state.result.value == 2
