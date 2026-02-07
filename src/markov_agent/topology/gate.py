from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState

StateT = TypeVar("StateT", bound=BaseState)


class ConfidenceGate(BaseModel):
    """A Gatekeeper that validates the state before allowing a transition.

    Implements Metric-Based Gating (Section 2 of the guide).
    """

    threshold: float = Field(
        default=0.8, description="Minimum confidence/value score required."
    )
    score_func: Callable[[Any], float] = Field(
        ..., description="Function to extract score from state."
    )
    fallback_node: str = Field(
        default="retry", description="Node to route to if threshold not met."
    )
    target_node: str = Field(..., description="Node to route to if threshold is met.")

    def route(self, state: Any) -> str:
        """Route based on the confidence score of the current state."""
        try:
            score = self.score_func(state)
        except Exception:
            score = 0.0

        if score >= self.threshold:
            return self.target_node
        return self.fallback_node
