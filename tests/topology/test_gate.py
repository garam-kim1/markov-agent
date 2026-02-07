from markov_agent.core.state import BaseState
from markov_agent.topology.gate import ConfidenceGate


class MockState(BaseState):
    confidence: float = 0.0

def test_confidence_gate_routing():
    gate = ConfidenceGate(
        threshold=0.8,
        score_func=lambda s: s.confidence,
        target_node="success",
        fallback_node="fail"
    )

    # Success case
    state_ok = MockState(confidence=0.9)
    assert gate.route(state_ok) == "success"

    # Failure case
    state_bad = MockState(confidence=0.5)
    assert gate.route(state_bad) == "fail"

    # Error case fallback
    assert gate.route(None) == "fail"
