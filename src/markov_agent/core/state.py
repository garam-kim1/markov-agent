from typing import Any, Self

from pydantic import BaseModel, Field

from markov_agent.core.probability import LogProb


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
        # Use model_dump + update + model_validate to ensure nested models are correctly handled
        data = self.model_dump()
        data.update(kwargs)
        return self.__class__.model_validate(data)

    def record_step(self, step_data: Any) -> None:
        """Append a snapshot or step data to history."""
        self.history.append(step_data)

    def get_markov_view(self) -> Any:
        """Return a view of the state that excludes history and meta.

        This enforces the Markov property by ensuring transitions only depend
        on the current state's explicitly defined fields.
        """
        # Create a proxy that excludes history and meta
        data = self.model_dump(exclude={"history", "meta"})

        class MarkovView:
            def __init__(self, d: dict[str, Any]) -> None:
                self.__dict__.update(d)

            def __repr__(self) -> str:
                return f"MarkovView({self.__dict__})"

        return MarkovView(data)

    def record_probability(
        self,
        source: str,
        target: str | None = None,
        probability: float = 1.0,
        distribution: dict[str, float] | None = None,
    ) -> None:
        """Record the probability of a chosen transition path."""
        if "path_probabilities" not in self.meta:
            self.meta["path_probabilities"] = []

        record = {"source": source, "probability": probability, "node": source}
        if target:
            record["target"] = target

        self.meta["path_probabilities"].append(record)

        # Shannon Entropy calculation: H = -sum(p * log2(p))
        if distribution:
            from markov_agent.core.probability import calculate_entropy

            entropy = calculate_entropy(distribution)
            if "step_entropy" not in self.meta:
                self.meta["step_entropy"] = []
            self.meta["step_entropy"].append(entropy)

        # Update overall confidence (joint probability of the trace)
        # Use log-space arithmetic to prevent underflow
        current_log_prob = self.meta.get("cumulative_log_prob", 0.0)  # log(1.0) = 0.0
        step_log_prob = LogProb.from_float(probability)
        new_log_prob = LogProb.multiply(current_log_prob, step_log_prob)

        self.meta["cumulative_log_prob"] = new_log_prob
        self.meta["confidence"] = LogProb.to_float(new_log_prob)
