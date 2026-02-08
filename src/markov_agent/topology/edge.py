import random
from collections.abc import Callable
from typing import Any

# Returns next_node_id OR a dict of {next_node_id: probability}
TransitionDistribution = dict[str, float]
TransitionResult = str | TransitionDistribution | None
RouterFunction = Callable[[Any], TransitionResult]


class Edge:
    """Defines a transition between nodes.

    Supports both deterministic (str) and probabilistic (TransitionDistribution) transitions.
    """

    def __init__(self, source: str, target_func: RouterFunction):
        self.source = source
        self.target_func = target_func

    def get_distribution(self, state: Any) -> TransitionDistribution:
        """Calculate the transition probability distribution."""
        result = self.target_func(state)

        if result is None:
            return {}

        if isinstance(result, str):
            return {result: 1.0}

        if isinstance(result, dict):
            weights = list(result.values())
            total_weight = sum(weights)
            if total_weight <= 0:
                return {}
            return {node: weight / total_weight for node, weight in result.items()}

        msg = f"Invalid transition result type: {type(result)}"
        raise ValueError(msg)

    def route(self, state: Any) -> tuple[str | None, float]:
        """Determine the next node ID and the transition probability.

        If the router returns a distribution, performs weighted random selection.
        Returns (next_node_id, probability).
        """
        distribution = self.get_distribution(state)

        if not distribution:
            return None, 1.0

        nodes = list(distribution.keys())
        weights = list(distribution.values())

        # Weighted random selection
        selected_node = random.choices(  # noqa: S311
            nodes, weights=weights, k=1
        )[0]

        # Find the probability of the selected node
        probability = distribution[selected_node]

        # Record probability in state if it's a Markov State
        if hasattr(state, "record_probability") and callable(state.record_probability):
            state.record_probability(
                self.source, probability, distribution=distribution
            )

        return selected_node, probability


class ProbabilisticEdge(Edge):
    """Explicitly named class for probabilistic edges, though Edge now supports both."""
