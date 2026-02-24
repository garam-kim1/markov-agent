import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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

    def __init__(
        self,
        source: str | None = None,
        target: str | RouterFunction | None = None,
        target_func: RouterFunction | None = None,
        condition: Callable[[Any], bool] | None = None,
        *,
        default: bool = False,
        weight: float = 1.0,
    ):
        self.source = source
        # Backward compatibility for Edge(source, target_func)
        if callable(target) and target_func is None:
            self.target_func = target
            self.target = None
        else:
            self.target = target if isinstance(target, str) else None
            self.target_func = target_func

        self.condition = condition
        self.default = default
        self.weight = weight

    def get_distribution(self, state: Any) -> TransitionDistribution:
        """Calculate the transition probability distribution."""
        # Trust the caller to provide the correct view (Graph handles strict_markov)
        view = state

        if self.target_func:
            return self._get_distribution_from_func(view)

        if self.target:
            return self._get_distribution_from_target(view)

        return {}

    def _get_distribution_from_func(self, view: Any) -> TransitionDistribution:
        func = self.target_func
        if not func:
            return {}

        func = cast("RouterFunction", func)
        result = func(view)

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

    def _get_distribution_from_target(self, view: Any) -> TransitionDistribution:
        if not self.target:
            return {}

        if self.condition:
            try:
                if self.condition(view):
                    return {self.target: self.weight}
            except Exception:
                return {}
        elif self.default or not self.condition:
            return {self.target: self.weight}

        return {}

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

    def __rshift__(self, other: Any) -> "Edge":
        """Support Edge >> Node syntax to set target."""
        from markov_agent.topology.node import BaseNode

        if isinstance(other, BaseNode):
            self.target = other.name
        elif isinstance(other, str):
            self.target = other
        else:
            msg = f"Cannot set target of Edge to {type(other)}"
            raise TypeError(msg)
        return self


class Flow(list):
    """A collection of edges with a pointer to the last node for chaining."""

    def __init__(self, edges: list[Edge] | None = None, last_node: Any = None) -> None:
        super().__init__(edges or [])
        self.last_node = last_node

    def __rshift__(self, other: Any) -> "Flow":
        """Link nodes or edges into a flow using the >> operator."""
        from markov_agent.topology.edge import Switch
        from markov_agent.topology.node import BaseNode

        if self.last_node is None:
            msg = "Cannot connect to a flow that has already been terminated (e.g. by a Switch)."
            raise TypeError(msg)

        # If last_node was an Edge and we are giving it a target
        if (
            isinstance(self.last_node, Edge)
            and not self.last_node.target
            and isinstance(other, (BaseNode, str))
        ):
            self.last_node.target = other.name if hasattr(other, "name") else other
            return Flow(list(self), last_node=other)

        source_name = (
            self.last_node.name
            if hasattr(self.last_node, "name")
            else str(self.last_node)
        )

        if isinstance(other, Switch):
            if not source_name:
                msg = "Cannot connect Switch to an incomplete flow."
                raise ValueError(msg)
            edges = []
            for condition, target in other.cases.items():
                target_name = target.name if hasattr(target, "name") else target
                edges.append(
                    Edge(source=source_name, target=target_name, condition=condition)
                )
            if other.default:
                target_name = (
                    other.default.name
                    if hasattr(other.default, "name")
                    else other.default
                )
                edges.append(Edge(source=source_name, target=target_name, default=True))
            return Flow([*list(self), *edges], last_node=None)

        if isinstance(other, (BaseNode, str)):
            target_name = other.name if hasattr(other, "name") else other
            new_edge = Edge(source=source_name, target=target_name)
            return Flow([*list(self), new_edge], last_node=other)

        if isinstance(other, Edge):
            other.source = source_name
            return Flow([*list(self), other], last_node=other)

        msg = f"Cannot link {source_name} with {type(other)}"
        raise TypeError(msg)


class ProbabilisticEdge(Edge):
    """Explicitly named class for probabilistic edges, though Edge now supports both."""


class Switch:
    """A fluent branching mechanism for the >> operator.

    Example:
        g.connect(start_node >> Switch({
            lambda s: s.score > 0.8: success_node,
            lambda s: s.score < 0.2: retry_node
        }, default=fallback_node))

    """

    def __init__(
        self, cases: dict[Callable[[Any], bool], Any], default: Any = None
    ) -> None:
        self.cases = cases
        self.default = default
