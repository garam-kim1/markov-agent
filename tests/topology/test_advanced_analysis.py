import numpy as np
import pytest

from markov_agent.core.probability import (
    apply_temperature,
    jensen_shannon_divergence,
    kl_divergence,
    normalize_distribution,
)
from markov_agent.topology.analysis import EmpiricalTransitionRecorder, TopologyAnalyzer
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class MockNode(BaseNode):
    async def execute(self, state):
        return state


def test_probability_primitives():
    # Normalization
    d1 = {"a": 1, "b": 1}
    assert normalize_distribution(d1) == {"a": 0.5, "b": 0.5}

    d2 = {"a": 0, "b": 0}
    assert normalize_distribution(d2) == {"a": 0.5, "b": 0.5}

    # Temperature
    dist = {"a": 0.1, "b": 0.9}
    # Greedy
    assert apply_temperature(dist, 0.0) == {"a": 0.0, "b": 1.0}
    # Uniform
    high_temp = apply_temperature(dist, 100.0)
    assert high_temp["a"] == pytest.approx(0.5, abs=0.1)

    # Divergence
    p = {"a": 1.0, "b": 0.0}
    q = {"a": 0.5, "b": 0.5}
    assert kl_divergence(p, q) > 0
    assert jensen_shannon_divergence(p, q) > 0
    assert jensen_shannon_divergence(p, q) == jensen_shannon_divergence(q, p)


def test_topology_diagnostics():
    # Create a simple ergodic graph: A -> B, B -> A
    nodes = {"A": MockNode(name="A"), "B": MockNode(name="B")}
    edges = [
        Edge(source="A", target_func=lambda _: "B"),
        Edge(source="B", target_func=lambda _: "A"),
    ]
    graph = Graph(name="test", nodes=nodes, edges=edges, entry_point="A")
    analyzer = TopologyAnalyzer(graph)

    matrix = analyzer.extract_matrix(sample_count=1)
    # Expected transition matrix is [[0, 1], [1, 0]]
    assert np.allclose(matrix, [[0, 1], [1, 0]])

    # This matrix is NOT ergodic because it is periodic (period 2)
    # Wait, my is_ergodic check for P^k might catch this.
    # P^2 = [[1, 0], [0, 1]], P^3 = [[0, 1], [1, 0]] ... never all positive.
    assert not analyzer.is_ergodic(matrix)

    # Add self-loops to make it aperiodic
    edges.append(Edge(source="A", target_func=lambda _: {"A": 0.5, "B": 0.5}))
    # Update matrix
    matrix = analyzer.extract_matrix(sample_count=10)
    # P should be approx [[0.5, 0.5], [1.0, 0.0]]
    assert analyzer.is_ergodic(matrix)

    mixing_time = analyzer.calculate_mixing_time(matrix)
    assert mixing_time > 0

    traj_prob = analyzer.simulate_trajectory_probability(["A", "B", "A"], matrix)
    assert 0 < traj_prob < 1


def test_empirical_recorder():
    recorder = EmpiricalTransitionRecorder(["A", "B"])
    recorder.observe_transition("A", "B")
    recorder.observe_transition("A", "B")
    recorder.observe_transition("B", "A")

    matrix = recorder.get_transition_matrix()
    # A -> B twice, A -> A zero: P[A, B] = 1.0, P[A, A] = 0.0
    # B -> A once, B -> B zero: P[B, A] = 1.0, P[B, B] = 0.0
    assert np.allclose(matrix, [[0, 1], [1, 0]])


def test_mermaid_generation():
    nodes = {"A": MockNode(name="A"), "B": MockNode(name="B")}
    edges = [Edge(source="A", target_func=lambda _: "B")]
    graph = Graph(name="test", nodes=nodes, edges=edges, entry_point="A")
    analyzer = TopologyAnalyzer(graph)
    matrix = analyzer.extract_matrix()

    mermaid = analyzer.generate_mermaid_graph(matrix)
    assert "graph TD" in mermaid
    assert 'A -- "1.00" --> B' in mermaid
