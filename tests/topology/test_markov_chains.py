import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.selectors import MajorityVoteSelector
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class MarkovState(BaseState):
    current_val: int = 0


class MockNode(BaseNode[MarkovState]):
    async def execute(self, state: MarkovState) -> MarkovState:
        state.current_val += 1
        return state


def test_probabilistic_edge_routing():
    # Test that the edge correctly samples from a distribution
    def router(state):
        return {"A": 0.0, "B": 1.0}  # 100% chance to go to B

    edge = Edge(source="START", target_func=router)

    # Run multiple times to verify distribution (deterministic here)
    for _ in range(10):
        next_node, prob, _ = edge.route({})
        assert next_node == "B"
        assert prob == 1.0

def test_probabilistic_edge_weighted_sampling():
    # Test weighted sampling
    def router(state):
        return {"A": 0.5, "B": 0.5}

    edge = Edge(source="START", target_func=router)

    results = []
    for _ in range(100):
        next_node, prob, _ = edge.route({})
        results.append(next_node)
        assert prob == 0.5

    count_a = results.count("A")
    count_b = results.count("B")

    # With 100 samples, 50/50 is likely within 30-70
    assert 20 < count_a < 80
    assert 20 < count_b < 80


@pytest.mark.asyncio
async def test_graph_records_probabilities():
    node_a = MockNode(name="A")
    node_b = MockNode(name="B")

    def router(state):
        if state.current_val == 1:
            return {"B": 1.0}
        return None

    graph = Graph(
        name="markov_graph",
        nodes={"A": node_a, "B": node_b},
        edges=[Edge(source="A", target_func=router)],
        entry_point="A",
        state_type=MarkovState,
    )

    state = MarkovState()
    final_state = await graph.run(state)

    assert final_state.current_val == 2
    assert "path_probabilities" in final_state.meta
    assert final_state.meta["path_probabilities"][0]["node"] == "A"
    assert final_state.meta["path_probabilities"][0]["probability"] == 1.0
    assert final_state.meta["confidence"] == 1.0


def test_majority_vote_selector():
    selector = MajorityVoteSelector()

    # Simple strings
    samples = ["apple", "banana", "apple", "cherry", "apple"]
    assert selector.select(samples) == "apple"

    # Dicts
    samples_dict = [{"v": 1}, {"v": 2}, {"v": 1}]
    assert selector.select(samples_dict) == {"v": 1}


@pytest.mark.asyncio
async def test_ppu_majority_selector_integration():
    config = ADKConfig(model_name="mock")

    # Mock responder that returns varied results to test majority vote
    call_count = 0

    def mock_responder(prompt):
        nonlocal call_count
        call_count += 1
        # 1st, 2nd, 3rd calls return "A", 4th, 5th return "B"
        if call_count <= 3:
            return "A"
        return "B"

    ppu = ProbabilisticNode(
        name="voter",
        adk_config=config,
        prompt_template="test",
        samples=5,
        selector="majority",
        mock_responder=mock_responder,
        state_type=MarkovState,
    )

    state = MarkovState()
    # We need to run it through a graph or call execute directly if it supports selector
    # ProbabilisticNode._execute_impl uses self.selector

    updated_state = await ppu.execute(state)
    # The result "A" should be selected (3 vs 2)
    # Note: ProbabilisticNode.execute calls parse_result which records history
    assert updated_state.history[-1]["output"] == "A"

    # Verify confidence recording
    # ppu.execute doesn't record probability in meta by default (only _run_async_impl does)
    # Let's test _run_async_impl via Graph
    call_count = 0  # Reset for the graph run

    graph = Graph(
        name="ppu_graph",
        nodes={"voter": ppu},
        edges=[],
        entry_point="voter",
        state_type=MarkovState,
    )

    state = MarkovState()
    final_state = await graph.run(state)

    # Check history
    assert final_state.history[-1]["output"] == "A"

    # Check meta from PPU execution
    # voter_ppu probability should be 3/5 = 0.6
    ppu_prob = next(
        p for p in final_state.meta["path_probabilities"] if p["node"] == "voter_ppu"
    )
    assert ppu_prob["probability"] == 0.6
    assert final_state.meta["confidence"] == 0.6
