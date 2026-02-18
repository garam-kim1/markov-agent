import json

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService, Session
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.token_utils import count_tokens


class DeepState(BaseState):
    nested_data: dict = Field(default_factory=dict)
    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append", "max_length": 5}
    )
    iteration: int = 0


def create_mock_ctx(state_dict, agent):
    session = Session(
        id="test_session", app_name="test_app", user_id="test_user", state=state_dict
    )
    return InvocationContext(
        session=session,
        session_service=InMemorySessionService(),
        invocation_id="test_inv",
        agent=agent,
        artifact_service=InMemoryArtifactService(),
    )


@pytest.mark.asyncio
async def test_recursive_state_growth_and_reduction():
    """Test a node that produces large outputs in a loop, forcing reduction every time."""

    # We use mock_responder to simulate LLM returning large state
    def mock_resp(prompt):
        return json.dumps(
            {
                "iteration": 1,
                "logs": ["Very long log entry " * 10],
                "nested_data": {"key": "data " * 50},
            }
        )

    config = ADKConfig(
        model_name="openai/mock-model",
        api_base="http://localhost:8080",
        max_input_tokens=200,  # Very tight budget
        compress_state=True,
    )

    node = ProbabilisticNode(
        name="stress_node",
        state_type=DeepState,
        adk_config=config,
        mock_responder=mock_resp,
        prompt_template="Current iteration: {{ iteration }}",
    )

    # Initial state
    state = DeepState(iteration=0)

    # Run multiple steps
    current_state_dict = state.model_dump()
    for i in range(3):
        ctx = create_mock_ctx(current_state_dict, node)
        async for _ in node._run_async_impl(ctx):
            pass

        session_state = ctx.session.state
        tokens = count_tokens(json.dumps(session_state))

        # It should be reduced
        assert config.max_input_tokens is not None
        assert tokens <= config.max_input_tokens + 150

        # Update iteration for next round
        session_state["iteration"] = i + 1
        current_state_dict = session_state


@pytest.mark.asyncio
async def test_parallel_state_interference():
    """Test if parallel sampling correctly handles state reduction without cross-interference."""

    def mock_resp(p):
        import random

        size = random.randint(10, 50)
        return json.dumps({"logs": [f"log_{size} " * size], "iteration": size})

    config = ADKConfig(
        model_name="openai/mock-model",
        api_base="http://localhost:8080",
        max_input_tokens=1000,
        compress_state=True,
        mock_responder=mock_resp,
    )

    node = ProbabilisticNode(
        name="parallel_stress",
        state_type=DeepState,
        adk_config=config,
        samples=3,
        prompt_template="Parallel test",
    )

    state = DeepState(logs=["Initial log"] * 10)
    ctx = create_mock_ctx(state.model_dump(), node)

    async for _ in node._run_async_impl(ctx):
        pass

    final_state = ctx.session.state
    # Reduction should have happened either in PPU or in individual samples
    assert count_tokens(json.dumps(final_state)) < 600
    assert "iteration" in final_state


@pytest.mark.asyncio
async def test_nested_non_serializable_state():
    """Test state containing objects that might cause issues during JSON serialization/token counting."""

    class CustomObj:
        def __repr__(self):
            return "<CustomObj>"

    config = ADKConfig(
        model_name="gemini-1.5-flash", compress_state=True, max_input_tokens=100
    )
    # mock_responder doesn't use the model, but it must be a valid registered name
    node = ProbabilisticNode(
        name="custom_obj_node",
        adk_config=config,
        prompt_template="test",
        mock_responder=lambda p: "{}",
    )

    state_dict = {"data": "A" * 500, "obj": CustomObj()}
    ctx = create_mock_ctx(state_dict, node)

    # This should not crash.
    async for _ in node._run_async_impl(ctx):
        pass

    # The data should be reduced
    assert count_tokens(json.dumps(ctx.session.state, default=str)) < 200
