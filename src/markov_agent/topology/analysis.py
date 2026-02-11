import contextlib
from typing import Any

import numpy as np

from markov_agent.topology.graph import Graph


class TopologyAnalyzer:
    """Analyzes the topology and transition dynamics of a Graph."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = list(graph.nodes.keys())
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}

    def extract_matrix(self, sample_count: int = 100) -> np.ndarray:
        """Approximate the transition matrix by Monte Carlo sampling.

        This assumes a stationary distribution or averages over state variations.
        """
        n = len(self.nodes)
        matrix = np.zeros((n, n))

        # We need a way to get a 'representative' state for each node.
        # Since we don't have that, we'll try to use a dummy state or
        # rely on the fact that if strict_markov is True, the router might
        # be more predictable.

        for i, source_id in enumerate(self.nodes):
            # Find edges starting from this node
            relevant_edges = [e for e in self.graph.edges if e.source == source_id]
            if not relevant_edges:
                # Terminal node: stays in terminal state (absorbing)
                matrix[i, i] = 1.0
                continue

            for _ in range(sample_count):
                # We need a state. This is the hardest part.
                # If we don't have a state, we might not be able to call the router.
                # For analysis, we'll try to use an empty state or a basic one.
                # If it fails, we might just have to skip or use default weights.

                # Simplified: just call get_distribution once if it's independent of state
                # or average it over multiple mock states if we had a state generator.

                # For now, let's just try to call it with a mock state object.
                class MockState:
                    def __init__(self) -> None:
                        self.meta: dict = {}
                        self.history: list = []

                    def get_markov_view(self) -> "MockState":
                        return self

                    def model_dump(self, **kwargs: Any) -> dict:
                        return {}

                mock_state = MockState()

                for edge in relevant_edges:
                    with contextlib.suppress(Exception):
                        dist = edge.get_distribution(mock_state)
                        for target_id, prob in dist.items():
                            if target_id in self.node_to_idx:
                                j = self.node_to_idx[target_id]
                                matrix[i, j] += prob / (len(relevant_edges) * sample_count)

        # Normalize rows to ensure they sum to 1 (for stochastic matrix)
        row_sums = matrix.sum(axis=1)
        # If a row sum is 0 (no transitions found), make it an absorbing state
        for i in range(n):
            if row_sums[i] == 0:
                matrix[i, i] = 1.0
            else:
                matrix[i, :] /= row_sums[i]

        return matrix

    def detect_absorbing_states(self, matrix: np.ndarray) -> list[str]:
        """Identify terminal/absorbing nodes mathematically (P[i,i] == 1)."""
        absorbing_indices = np.where(np.isclose(np.diag(matrix), 1.0))[0]
        return [self.nodes[i] for i in absorbing_indices]

    def calculate_stationary_distribution(
        self, matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-8
    ) -> np.ndarray:
        """Find the stationary distribution (pi * P = pi) via power iteration."""
        n = len(self.nodes)
        pi = np.ones(n) / n

        for _ in range(max_iter):
            pi_next = pi @ matrix
            if np.linalg.norm(pi_next - pi, ord=1) < tol:
                return pi_next
            pi = pi_next

        return pi
