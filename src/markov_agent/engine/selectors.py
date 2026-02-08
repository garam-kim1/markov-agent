import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class BaseSelector[T](ABC):
    """Base class for sample selectors."""

    @abstractmethod
    def select(self, samples: list[T]) -> T:
        """Select the best sample from a list of samples."""


class MajorityVoteSelector[T](BaseSelector[T]):
    """Selects the most frequent sample (Majority Voting)."""

    def select(self, samples: list[T]) -> T:
        if not samples:
            msg = "Cannot select from empty samples."
            raise ValueError(msg)

        # Count occurrences
        counts = {}
        for sample in samples:
            # Handle non-hashable types (like dicts or Pydantic models)
            if isinstance(sample, BaseModel):
                key = sample.model_dump_json()
            elif isinstance(sample, (dict, list)):
                key = json.dumps(sample, sort_keys=True)
            else:
                key = sample

            counts[key] = counts.get(key, 0) + 1

        # Find the max
        best_key = max(counts, key=lambda k: counts[k])

        # Return the original sample that matches this key
        for sample in samples:
            if isinstance(sample, BaseModel):
                if sample.model_dump_json() == best_key:
                    return sample
            elif isinstance(sample, (dict, list)):
                if json.dumps(sample, sort_keys=True) == best_key:
                    return sample
            elif sample == best_key:
                return sample

        return samples[0]


class LLMJudgeSelector[T](BaseSelector[T]):
    """Uses an LLM to judge and select the best response."""

    def __init__(self, controller: Any, criteria: str = "accuracy and clarity"):
        self.controller = controller
        self.criteria = criteria

    def select(self, samples: list[T]) -> T:
        # This one is tricky because it's usually async.
        # But selectors in PPU are currently called in a way that could be sync or async.
        # The PPU's execute_parallel_sampling expects a selector_func.
        # If it's an LLM judge, we might need a specialized node or handle it in PPU.

        if len(samples) <= 1:
            return samples[0]

        # For now, we'll implement a simple heuristic or a placeholder
        # because the PPU's _verify_results already has some verification logic.
        # In a real implementation, this would trigger a judge prompt.

        return samples[0]


# Mapping of aliases to selector instances/classes
SELECTOR_REGISTRY = {
    "majority": MajorityVoteSelector(),
}
