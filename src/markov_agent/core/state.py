from typing import Any, Self

from pydantic import BaseModel, Field


class BaseState(BaseModel):
    """The base state object. State is the only source of truth.

    All specific application states should inherit from this.
    """

    history: list[Any] = Field(
        default_factory=list,
        description="Immutable history tracking for Time Travel Debugging.",
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata tracking for the current state (e.g. confidence, probabilities).",
    )

    def update(self, **kwargs: Any) -> Self:
        """Return a new instance of the state with updated fields."""
        return self.model_copy(update=kwargs)

    def record_step(self, step_data: Any) -> None:
        """Append a snapshot or step data to history."""
        self.history.append(step_data)

    def record_probability(
        self,
        node: str,
        probability: float,
        distribution: dict[str, float] | None = None,
    ) -> None:
        """Record the probability of a chosen transition path."""
        if "path_probabilities" not in self.meta:
            self.meta["path_probabilities"] = []
        self.meta["path_probabilities"].append(
            {"node": node, "probability": probability}
        )

        # Shannon Entropy calculation: H = -sum(p * log2(p))
        if distribution:
            import math

            entropy = -sum(p * math.log2(p) for p in distribution.values() if p > 0)
            if "step_entropy" not in self.meta:
                self.meta["step_entropy"] = []
            self.meta["step_entropy"].append(entropy)

        # Update overall confidence (joint probability of the trace)
        current_conf = self.meta.get("confidence", 1.0)
        self.meta["confidence"] = current_conf * probability
