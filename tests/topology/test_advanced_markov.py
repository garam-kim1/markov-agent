import math

import pytest

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class MarkovState(BaseState):
    val: int = 0


class IncrementNode(BaseNode[MarkovState]):
    async def execute(self, state: MarkovState) -> MarkovState:
        state.val += 1
        return state


@pytest.mark.asyncio
async def test_entropy_tracking():
    def router(_state):
        # Distribution probabilities: 0.5, 0.25, 0.25
        # We calculate the Shannon Entropy here
        # Result should be exactly 1.5
        return {"A": 0.5, "B": 0.25, "C": 0.25}

    edge = Edge(source="START", target_func=router)
    state = MarkovState()

    # route() no longer mutates state, so we manually record for the test
    result = edge.route(state)
    state.record_probability(
        source="START",
        target=result.next_node,
        probability=result.probability,
        distribution=result.distribution,
    )
    assert "step_entropy" in state.meta
    assert len(state.meta["step_entropy"]) == 1
    assert math.isclose(state.meta["step_entropy"][0], 1.5)


@pytest.mark.asyncio
async def test_run_beam():
    # Graph:
    # START -> (A: 0.8, B: 0.2)
    # A -> END (1.0)
    # B -> END (1.0)

    node_start = IncrementNode(name="START", state_type=MarkovState)
    node_a = IncrementNode(name="A", state_type=MarkovState)
    node_b = IncrementNode(name="B", state_type=MarkovState)
    node_end = IncrementNode(name="END", state_type=MarkovState)

    def start_router(state):
        return {"A": 0.8, "B": 0.2}

    def a_router(state):
        return {"END": 1.0}

    def b_router(state):
        return {"END": 1.0}

    graph = Graph(
        name="beam_graph",
        nodes={"START": node_start, "A": node_a, "B": node_b, "END": node_end},
        edges=[
            Edge(source="START", target_func=start_router),
            Edge(source="A", target_func=a_router),
            Edge(source="B", target_func=b_router),
        ],
        entry_point="START",
        state_type=MarkovState,
    )

    state = MarkovState()
    results = await graph.run_beam(state, width=2)

    assert len(results) == 2
    # The first result should be path through A (higher probability)
    # START(inc) -> start_router -> A(inc) -> a_router -> END(inc)
    # START(0.8) -> A(1.0) -> END
    # Conf = 0.8 * 1.0 = 0.8
    assert results[0].meta["confidence"] == 0.8
    assert results[0].val == 3  # START, A, END all incremented

    # Second result should be path through B
    # START(0.2) -> B(1.0) -> END
    # Conf = 0.2 * 1.0 = 0.2
    assert results[1].meta["confidence"] == 0.2
    assert results[1].val == 3


@pytest.mark.asyncio
async def test_run_beam_pruning():
    # START -> (A: 0.6, B: 0.3, C: 0.1)
    # width parameter is 2

    node_start = IncrementNode(name="START", state_type=MarkovState)
    node_a = IncrementNode(name="A", state_type=MarkovState)
    node_b = IncrementNode(name="B", state_type=MarkovState)
    node_c = IncrementNode(name="C", state_type=MarkovState)

    def start_router(state):
        return {"A": 0.6, "B": 0.3, "C": 0.1}

    graph = Graph(
        name="beam_graph_prune",
        nodes={"START": node_start, "A": node_a, "B": node_b, "C": node_c},
        edges=[Edge(source="START", target_func=start_router)],
        entry_point="START",
        state_type=MarkovState,
    )

    state = MarkovState()
    results = await graph.run_beam(state, width=2)

    assert len(results) == 2
    assert results[0].meta["confidence"] == 0.6
    assert results[1].meta["confidence"] == 0.3
