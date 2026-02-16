from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markov_agent.simulation.runner import SimulationResult
    from markov_agent.topology.edge import Edge
    from markov_agent.topology.graph import Graph


class TopologyOptimizer:
    """Optimizes the Graph structure based on simulation data and rewards.

    Implements 'Structural Autopoiesis' by rewriting graph components.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def prune_edges(
        self, results: list[SimulationResult], threshold: float = 0.1
    ) -> int:
        """Remove edges that are rarely used or lead to failure.

        Returns the number of edges removed.
        """
        # 1. Build empirical usage map
        edge_usage: dict[tuple[str, str], int] = {}
        edge_success: dict[tuple[str, str], int] = {}

        for res in results:
            traj = res.trajectory
            for i in range(len(traj) - 1):
                src = traj[i].get("node")
                dst = traj[i + 1].get("node")
                if src and dst:
                    edge = (src, dst)
                    edge_usage[edge] = edge_usage.get(edge, 0) + 1
                    if res.success:
                        edge_success[edge] = edge_success.get(edge, 0) + 1

        # 2. Identify edges to prune
        to_remove: list[Edge] = []
        for edge_obj in self.graph.edges:
            if not edge_obj.target or not edge_obj.source:
                continue

            pair = (edge_obj.source, edge_obj.target)
            usage = edge_usage.get(pair, 0)
            success_rate = edge_success.get(pair, 0) / usage if usage > 0 else 0.0

            # Prune if usage is above threshold OR success rate is terrible
            if usage > 0 and success_rate < threshold:
                to_remove.append(edge_obj)

        # 3. Apply changes
        count = 0
        for edge_obj in to_remove:
            if edge_obj in self.graph.edges:
                self.graph.edges.remove(edge_obj)
                count += 1

        return count

    def suggest_fission(
        self, results: list[SimulationResult], entropy_threshold: float = 1.5
    ) -> list[str]:
        """Identify nodes with high entropy that are candidates for splitting.

        High entropy suggests the LLM is 'confused' at this decision point.
        """
        node_entropies: dict[str, list[float]] = {}

        for res in results:
            # Check meta for entropy if available
            if hasattr(res.final_state, "meta"):
                step_entropies = res.final_state.meta.get("step_entropy", [])
                traj = res.trajectory
                for i, entropy in enumerate(step_entropies):
                    if i < len(traj):
                        node_id = traj[i].get("node")
                        if node_id:
                            if node_id not in node_entropies:
                                node_entropies[node_id] = []
                            node_entropies[node_id].append(entropy)

        candidates = []
        for node_id, entropies in node_entropies.items():
            avg_entropy = sum(entropies) / len(entropies)
            if avg_entropy > entropy_threshold:
                candidates.append(node_id)

        return candidates

    def apply_fission(self, node_id: str, new_node_names: list[str]) -> bool:
        """Split a node into multiple specialized nodes.

        Note: This is a complex operation that requires human or 'Meta-Agent'
        intervention to redefine the prompts for the new nodes.
        For now, we just clone the node structure to demonstrate the topology change.
        """
        if node_id not in self.graph.nodes:
            return False

        original_node = self.graph.nodes[node_id]

        for new_name in new_node_names:
            new_node = copy.deepcopy(original_node)
            new_node.name = new_name
            self.graph.add_node(new_node)

            # Redistribute edges: incoming to node_id now also go to new_name?
            # Or we need a gate. This is context-dependent.
            # Simplified: incoming edges to node_id are cloned for new_name
            for edge in list(self.graph.edges):
                if edge.target == node_id:
                    new_edge = copy.deepcopy(edge)
                    new_edge.target = new_name
                    self.graph.edges.append(new_edge)

                if edge.source == node_id:
                    new_edge = copy.deepcopy(edge)
                    new_edge.source = new_name
                    self.graph.edges.append(new_edge)

        return True

    def learn_from_rewards(
        self, results: list[SimulationResult], learning_rate: float = 0.1
    ) -> dict[tuple[str, str], float]:
        """Update transition probabilities based on cumulative rewards.

        This implements 'MDP for Evolution' by updating edge weights.
        """
        # 1. Calculate average reward per edge
        edge_rewards: dict[tuple[str, str], list[float]] = {}

        for res in results:
            traj = res.trajectory
            # reward could be in res.final_state.reward if it's a BaseState
            reward = getattr(res.final_state, "reward", 0.0)
            if not reward and hasattr(res.final_state, "meta"):
                reward = res.final_state.meta.get("reward", 0.0)

            for i in range(len(traj) - 1):
                src = traj[i].get("node")
                dst = traj[i + 1].get("node")
                if src and dst:
                    edge = (src, dst)
                    if edge not in edge_rewards:
                        edge_rewards[edge] = []
                    edge_rewards[edge].append(reward)

        avg_rewards = {edge: sum(r) / len(r) for edge, r in edge_rewards.items() if r}

        # 2. Update Edge distributions
        updates = {}
        for edge_obj in self.graph.edges:
            if edge_obj.target_func:
                continue

            if not edge_obj.source or not edge_obj.target:
                continue

            pair = (edge_obj.source, edge_obj.target)
            if pair in avg_rewards:
                reward_signal = avg_rewards[pair]
                # Adjust weight using learning rate and reward signal
                # New weight is calculated as old_weight * (1 + learning_rate * reward)
                # Ensure weight doesn't go below a small epsilon
                old_weight = getattr(edge_obj, "weight", 1.0)
                new_weight = max(
                    0.01, old_weight * (1.0 + learning_rate * reward_signal)
                )
                edge_obj.weight = new_weight
                updates[pair] = new_weight

        return updates
