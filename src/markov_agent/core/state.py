from typing import Annotated, Any, Self, TypeVar

from pydantic import BaseModel, Field

from markov_agent.core.probability import LogProb

T = TypeVar("T")
AppendList = Annotated[list[T], Field(json_schema_extra={"behavior": "append"})]
AppendString = Annotated[str, Field(json_schema_extra={"behavior": "append"})]


class BaseState(BaseModel):
    """The base state object. State is the only source of truth.

    All specific application states should inherit from this.
    """

    history: list[Any] = Field(
        default_factory=list,
        json_schema_extra={"behavior": "append", "max_length": 10},
        description="Immutable history tracking for Time Travel Debugging.",
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata tracking for the current state (e.g. confidence, probabilities).",
    )

    reward: float = Field(
        default=0.0,
        description="Cumulative reward for the current trajectory.",
    )

    @property
    def reasoning(self) -> str | None:
        """Return the reasoning/thought process from the last LLM invocation if available."""
        return self.meta.get("reasoning")

    def update(self, **kwargs: Any) -> Self:
        """Return a new instance of the state with updated fields.

        Supports 'behavior="append"' and 'max_length=N' in Field metadata.
        If a field is marked with behavior="append", new values will be
        appended to lists or concatenated to strings instead of replacing them.
        If max_length is provided, lists will be capped to the last N items.
        """
        updates = {}
        for key, value in kwargs.items():
            field = self.__class__.model_fields.get(key)
            behavior = None
            max_length = None
            if field and field.json_schema_extra:
                behavior = field.json_schema_extra.get("behavior")
                max_length = field.json_schema_extra.get("max_length")

            if behavior == "append" and hasattr(self, key):
                current_val = getattr(self, key)
                if isinstance(current_val, list):
                    if isinstance(value, list):
                        new_val = current_val + value
                    else:
                        new_val = [*current_val, value]

                    if max_length and isinstance(max_length, int):
                        new_val = new_val[-max_length:]
                    updates[key] = new_val
                elif isinstance(current_val, str):
                    new_val = (current_val or "") + str(value)
                    if max_length and isinstance(max_length, int):
                        new_val = new_val[-max_length:]
                    updates[key] = new_val
                else:
                    updates[key] = value
            else:
                updates[key] = value

        # Use model_copy for performance.
        return self.model_copy(update=updates, deep=True)

    def record_step(self, step_data: Any) -> None:
        """Append a snapshot or step data to history.

        Respects 'max_history' if set in meta, or 'max_length' in field metadata.
        Defaults to 10.
        """
        self.history.append(step_data)

        # 1. Check meta override
        max_history = self.meta.get("max_history")

        # 2. Check field metadata
        if max_history is None:
            field = self.__class__.model_fields.get("history")
            if field and field.json_schema_extra:
                max_history = field.json_schema_extra.get("max_length")

        # 3. Default
        if max_history is None:
            max_history = 10

        if (
            max_history
            and isinstance(max_history, int)
            and len(self.history) > max_history
        ):
            self.history = self.history[-max_history:]

    def compress(self, max_tokens: int, model_name: str = "gpt-3.5-turbo") -> Self:
        """Compress the state to fit within max_tokens."""
        from markov_agent.engine.token_utils import reduce_dict_to_tokens

        state_dict = self.model_dump()
        reduced_dict = reduce_dict_to_tokens(state_dict, max_tokens, model_name)
        return self.__class__.model_validate(reduced_dict)

    def record_reward(self, amount: float) -> None:
        """Add to the cumulative reward."""
        self.reward += amount

    def get_markov_view(self) -> Any:
        """Return a view of the state that excludes history and meta.

        This enforces the Markov property by ensuring transitions only depend
        on the current state's explicitly defined fields.
        """
        # Create a proxy that excludes history and meta
        # We don't use model_dump because it's recursive and converts nested models to dicts
        exclude = {"history", "meta"}
        data = {k: v for k, v in self.__dict__.items() if k not in exclude}

        class MarkovView:
            def __init__(self, d: dict[str, Any]) -> None:
                self.__dict__.update(d)

            def get(self, key: str, default: Any = None) -> Any:
                return self.__dict__.get(key, default)

            def __repr__(self) -> str:
                return f"MarkovView({self.__dict__})"

        return MarkovView(data)

    def to_dict(
        self, *, exclude_history: bool = True, exclude_meta: bool = True
    ) -> dict[str, Any]:
        """Convert state to a dictionary, optionally excluding history and meta."""
        exclude = set()
        if exclude_history:
            exclude.add("history")
        if exclude_meta:
            exclude.add("meta")
        return self.model_dump(exclude=exclude)

    def to_json(
        self, *, exclude_history: bool = True, exclude_meta: bool = True, **kwargs: Any
    ) -> str:
        """Convert state to a JSON string, optionally excluding history and meta."""
        exclude = set()
        if exclude_history:
            exclude.add("history")
        if exclude_meta:
            exclude.add("meta")
        return self.model_dump_json(exclude=exclude, **kwargs)

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

    def save(self, path: str) -> None:
        """Save the state to a JSON file."""
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> Self:
        """Load the state from a JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text())
