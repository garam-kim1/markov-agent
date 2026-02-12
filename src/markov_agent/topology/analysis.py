import contextlib
from collections.abc import Callable

import numpy as np

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph


class TopologyAnalyzer:
    """Analyzes the topology and transition dynamics of a Graph."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = list(graph.nodes.keys())
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}

    def extract_matrix(
        self,
        sample_count: int = 100,
        state_sampler: Callable[[], BaseState] | None = None,
    ) -> np.ndarray:
        """Approximate the transition matrix by sampling or structural analysis.

        If state_sampler is provided, it generates real states to test transitions.
        Otherwise, it performs structural analysis by inspecting edge definitions.

        Args:
            sample_count: Number of states to sample per node.
            state_sampler: Function that returns a BaseState instance.

        """
        n = len(self.nodes)
        matrix = np.zeros((n, n))

        for i, source_id in enumerate(self.nodes):
            relevant_edges = [e for e in self.graph.edges if e.source == source_id]
            if not relevant_edges:
                matrix[i, i] = 1.0  # Absorbing
                continue

            for _ in range(sample_count):
                state = state_sampler() if state_sampler else BaseState()

                for edge in relevant_edges:
                    with contextlib.suppress(Exception):
                        dist = edge.get_distribution(state)
                        for target_id, prob in dist.items():
                            if target_id in self.node_to_idx:
                                j = self.node_to_idx[target_id]
                                # Average probability across all edges and samples
                                matrix[i, j] += prob / (len(relevant_edges) * sample_count)

        # Normalize rows to ensure they sum to 1 (for stochastic matrix)
        row_sums = matrix.sum(axis=1)
        for i in range(n):
            if row_sums[i] == 0:
                matrix[i, i] = 1.0
            else:
                matrix[i, :] /= row_sums[i]

        return matrix

    def detect_absorbing_states(self, matrix: np.ndarray) -> list[str]:
        """Identify terminal/absorbing nodes mathematically (P[i,i] == 1)."""
        absorbing_indices = np.where(np.isclose(np.diag(matrix), 1.0))[0]
        # Ensure it's truly absorbing (no transitions to other states)
        return [
            self.nodes[idx]
            for idx in absorbing_indices
            if np.isclose(matrix[idx, idx], 1.0)
            and np.allclose(np.delete(matrix[idx, :], idx), 0.0)
        ]

    def calculate_stationary_distribution(
        self, matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-8
    ) -> np.ndarray:
        """Find the stationary distribution (pi * P = pi) via power iteration.

        Formula: pi_{t+1} = pi_t * P
        """
        n = len(self.nodes)
        pi = np.ones(n) / n

        for _ in range(max_iter):
            pi_next = pi @ matrix
            if np.linalg.norm(pi_next - pi, ord=1) < tol:
                return pi_next
            pi = pi_next

        return pi

    def is_ergodic(self, matrix: np.ndarray) -> bool:
        """Check if the Markov Chain is ergodic (irreducible and aperiodic).

        A chain is ergodic if it is possible to go from every state to every other
        state (irreducible) and the return times to any state are not restricted
        to multiples of some integer > 1 (aperiodic).
        """
        n = len(matrix)
        # Check irreducibility: (I + P)^(n-1) > 0
        id_plus_p = np.eye(n) + matrix
        reachability = np.linalg.matrix_power(id_plus_p, n - 1)
        if not np.all(reachability > 0):
            return False

        # Check aperiodicity: If primitive (P^k > 0 for some k), then it's ergodic.
        # For an irreducible matrix, it's aperiodic if at least one diagonal element is > 0
        # or if we check the GCD of cycle lengths. A simpler check for many cases:
        if np.any(np.diag(matrix) > 0):
            return True

        # More robust check: P^k should have all positive entries for large k
        pk = np.linalg.matrix_power(matrix, n * n)
        return bool(np.all(pk > 0))

    def calculate_mixing_time(self, matrix: np.ndarray, tol: float = 1e-4) -> int:
        """Calculate the number of steps to approach stationary distribution.

        Uses the second largest eigenvalue modulus (SLEM) or power iteration.
        Formula: t_mix(tol) approx log(1/tol) / log(1/|lambda_2|)
        """
        pi = self.calculate_stationary_distribution(matrix)
        n = len(self.nodes)
        v = np.ones(n) / n

        for t in range(1, 1000):
            v = v @ matrix
            if np.linalg.norm(v - pi, ord=1) < tol:
                return t
        return 1000

    def simulate_trajectory_probability(
        self, trajectory: list[str], matrix: np.ndarray
    ) -> float:
        """Calculate the likelihood of a specific sequence of nodes.

        Formula: P(T) = P(s_0) * product_{t=1}^L P(s_t | s_{t-1})
        Assumes uniform starting probability if not in an absorbing state.
        """
        if not trajectory:
            return 0.0

        prob = 1.0 / len(self.nodes)  # Initial state probability (uniform assumption)
        for i in range(len(trajectory) - 1):
            src = trajectory[i]
            dst = trajectory[i+1]
            if src not in self.node_to_idx or dst not in self.node_to_idx:
                return 0.0
            prob *= matrix[self.node_to_idx[src], self.node_to_idx[dst]]

        return prob

    def generate_mermaid_graph(
        self, matrix: np.ndarray, threshold: float = 0.01
    ) -> str:
        """Generate a Mermaid.js diagram of the transition matrix.

        Args:
            matrix: Transition matrix.
            threshold: Minimum probability to display an edge.

        """
        lines = ["graph TD"]
        for i, source in enumerate(self.nodes):
            for j, target in enumerate(self.nodes):
                prob = matrix[i, j]
                if prob > threshold:
                    # Style based on probability
                    weight = ""
                    if prob > 0.8:
                        weight = ":::bold"
                    lines.append(f'    {source} -- "{prob:.2f}" --> {target}{weight}')

        return "\n".join(lines)


class EmpiricalTransitionRecorder:
    """Records actual runtime transitions to estimate the empirical matrix."""

    def __init__(self, nodes: list[str]):
        self.nodes = nodes
        self.node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
        self.counts = np.zeros((len(nodes), len(nodes)))

    def observe_transition(self, source: str, target: str) -> None:
        """Record a transition between two nodes."""
        if source in self.node_to_idx and target in self.node_to_idx:
            i = self.node_to_idx[source]
            j = self.node_to_idx[target]
            self.counts[i, j] += 1

    def get_transition_matrix(self) -> np.ndarray:
        """Return the estimated transition matrix P_ij = C_ij / sum_k C_ik."""
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        row_sums = self.counts.sum(axis=1)

        for i in range(n):
            if row_sums[i] == 0:
                matrix[i, i] = 1.0
            else:
                matrix[i, :] = self.counts[i, :] / row_sums[i]

        return matrix
