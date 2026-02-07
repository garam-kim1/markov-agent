import pytest
from markov_agent.engine.mcts import MCTSNode
from markov_agent.engine.eig import EntropyCheck
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.core.state import BaseState
import json

class MockState(BaseState):
    val: str = ""
    entropy_score: float = 0.0
    next_step: str = ""

@pytest.mark.asyncio
async def test_entropy_check_logic():
    cfg = ADKConfig(model_name="mock")
    # Mock responder to return a specific entropy score
    def mock_resp(p):
        return "0.8"
    
    node = EntropyCheck(
        name="test_eig",
        adk_config=cfg,
        threshold=0.5,
        state_type=MockState,
        mock_responder=mock_resp
    )
    
    state = MockState(val="test")
    new_state = await node.execute(state)
    
    assert new_state.entropy_score == 0.8
    assert new_state.next_step == "clarification"

@pytest.mark.asyncio
async def test_mcts_node_execution():
    cfg = ADKConfig(model_name="mock")
    
    def mock_resp(p):
        if "Evaluate" in p:
            return "0.9"
        if "Generate" in p:
            return json.dumps([{"val": "branch1"}, {"val": "branch2"}])
        return "default"

    node = MCTSNode(
        name="test_mcts",
        adk_config=cfg,
        max_rollouts=2,
        expansion_k=2,
        state_type=MockState,
        mock_responder=mock_resp
    )
    
    state = MockState(val="start")
    new_state = await node.execute(state)
    
    assert new_state.val in ["branch1", "branch2"]
