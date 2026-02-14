import json

import pytest
from pydantic import Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.eig import EntropyCheck
from markov_agent.engine.mcts import MCTSNode


class SearchState(BaseState):
    value: int = 0
    history: list[str] = Field(default_factory=list)
    entropy_score: float = 0.0
    next_step: str = "searching"


@pytest.mark.asyncio
async def test_mcts_deep_search():
    """Test that MCTS selects the best branch based on simulated rewards."""
    cfg = ADKConfig(model_name="mock")

    def mock_responder(prompt: str) -> str:
        # Prompt contains state info
        if "Evaluate" in prompt:
            # Reward: 1.0 if value is 10, 0.1 otherwise
            if '"value": 10' in prompt:
                return "1.0"
            return "0.1"

        if "Generate" in prompt:
            # Generate two branches: one good, one bad
            return json.dumps(
                [{"value": 10, "history": ["good"]}, {"value": 1, "history": ["bad"]}]
            )
        return "0"

    node = MCTSNode(
        name="deep_mcts",
        adk_config=cfg,
        max_rollouts=10,
        expansion_k=2,
        state_type=SearchState,
        mock_responder=mock_responder,
    )

    state = SearchState(value=0)
    final_state = await node.execute(state)

    # MCTS should have converged on the branch with value=10
    assert final_state.value == 10
    assert "good" in final_state.history


@pytest.mark.asyncio
async def test_eig_threshold_logic():
    """Test EIG logic for triggering clarification."""
    cfg = ADKConfig(model_name="mock")

    # High entropy case
    def mock_high(p):
        return "0.9"

    node_high = EntropyCheck(
        name="eig_high",
        adk_config=cfg,
        threshold=0.5,
        state_type=SearchState,
        mock_responder=mock_high,
    )

    state = SearchState()
    res_high = await node_high.execute(state)
    assert res_high.next_step == "clarification"
    assert res_high.entropy_score == 0.9

    # Low entropy case
    def mock_low(p):
        return "0.2"

    node_low = EntropyCheck(
        name="eig_low",
        adk_config=cfg,
        threshold=0.5,
        state_type=SearchState,
        mock_responder=mock_low,
    )

    res_low = await node_low.execute(state)
    assert res_low.next_step == "execution"
    assert res_low.entropy_score == 0.2
