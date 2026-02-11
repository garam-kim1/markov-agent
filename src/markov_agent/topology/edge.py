import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from markov_agent.core.probability import LogProb, calculate_entropy

# Returns next_node_id OR a dict of {next_node_id: probability}
TransitionDistribution = dict[str, float]
RouterFunction = Callable[[Any], str | TransitionDistribution | None]


@dataclass
class TransitionResult:
    """The result of a routing operation."""

    next_node: str | None
    probability: float
    distribution: dict[str, float]
    entropy: float
    log_prob: float


class Edge:
    """Defines a transition between nodes.

    Supports both deterministic (str) and probabilistic (TransitionDistribution) transitions.
    """

    def __init__(self, source: str, target_func: RouterFunction):
        self.source = source
        self.target_func = target_func

    def get_distribution(self, state: Any) -> TransitionDistribution:
        """Calculate the transition probability distribution."""
        # Check if state is a MarkovView or similar
        view = state.get_markov_view() if hasattr(state, "get_markov_view") else state
        result = self.target_func(view)

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

    def route(self, state: Any) -> TransitionResult:
        """Determine the next node ID and the transition probability.

        If the router returns a distribution, performs weighted random selection.
        """
        distribution = self.get_distribution(state)

        if not distribution:
            return TransitionResult(
                next_node=None,
                probability=1.0,
                distribution={},
                entropy=0.0,
                log_prob=0.0,
            )

        nodes = list(distribution.keys())
        weights = list(distribution.values())

        # Weighted random selection
        selected_node = random.choices(  # noqa: S311
            nodes, weights=weights, k=1
        )[0]

        # Find the probability of the selected node
        probability = distribution[selected_node]

        entropy = calculate_entropy(distribution)
        log_prob = LogProb.from_float(probability)

        return TransitionResult(
            next_node=selected_node,
            probability=probability,
            distribution=distribution,
            entropy=entropy,
            log_prob=log_prob,
        )


class ProbabilisticEdge(Edge):
    """Explicitly named class for probabilistic edges, though Edge now supports both."""
