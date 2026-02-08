import random
from collections.abc import Callable
from typing import Any, Union

# Returns next_node_id OR a dict of {next_node_id: probability}
TransitionDistribution = dict[str, float]
TransitionResult = Union[str, TransitionDistribution, None]
RouterFunction = Callable[[Any], TransitionResult]


class Edge:
    """Defines a transition between nodes.

    Supports both deterministic (str) and probabilistic (TransitionDistribution) transitions.
    """

    def __init__(self, source: str, target_func: RouterFunction):
        self.source = source
        self.target_func = target_func

    def route(self, state: Any) -> tuple[str | None, float]:
        """Determines the next node ID and the transition probability.

        If the router returns a distribution, performs weighted random selection.
        Returns (next_node_id, probability).
        """
        result = self.target_func(state)

        if result is None:
            return None, 1.0

        if isinstance(result, str):
            return result, 1.0

        if isinstance(result, dict):
            # Markov Transition Logic
            nodes = list(result.keys())
            weights = list(result.values())
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(weights)
            if total_weight <= 0:
                return None, 0.0
                
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted random selection
            if not nodes:
                return None, 1.0
                
            selected_node = random.choices(nodes, weights=normalized_weights, k=1)[0]
            
            # Find the probability of the selected node
            index = nodes.index(selected_node)
            probability = normalized_weights[index]
            
            return selected_node, probability

        raise ValueError(f"Invalid transition result type: {type(result)}")


class ProbabilisticEdge(Edge):
    """Explicitly named class for probabilistic edges, though Edge now supports both."""
    pass