from unittest.mock import AsyncMock, MagicMock

import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode


class MockState(BaseState):
    val: str = ""
    verified: bool = False

@pytest.mark.asyncio
async def test_ppu_verifier_selection():
    cfg = ADKConfig(model_name="mock")

    # Mock Verifier Node
    verifier = MagicMock()
    # Mock the execute method to return a state with verified=True for a specific candidate
    async def v_exec(state):
        if state.get("candidate") == "good":
            return {"verified": True}
        return {"verified": False}
    verifier.execute = v_exec

    node = ProbabilisticNode(
        name="test_ppu",
        adk_config=cfg,
        prompt_template="test",
        samples=2,
        verifier_node=verifier,
        state_type=MockState
    )

    # Mock the controller's parallel generation
    # We need to mock execute_parallel_sampling or controller.generate
    # Simplest is to mock controller.generate and use uniform sampling
    node.controller.generate = AsyncMock(side_effect=["bad", "good"])

    # Mock ADK Context
    ctx = MagicMock()
    ctx.session.state = {"val": "start"}

    # Run the internal impl
    async for _ in node._run_async_impl(ctx):
        pass

    # The result in session state should be "good" because verifier selected it
    assert ctx.session.state["history"][-1]["output"] == "good"
