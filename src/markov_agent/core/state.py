from typing import Any

from pydantic import BaseModel, Field


class BaseState(BaseModel):
    """
    The base state object. State is the only source of truth.
    All specific application states should inherit from this.
    """

    history: list[Any] = Field(
        default_factory=list,
        description="Immutable history tracking for Time Travel Debugging.",
    )

    def update(self, **kwargs):
        """
        Returns a new instance of the state with updated fields.
        """
        return self.model_copy(update=kwargs)

    def record_step(self, step_data: Any):
        """
        Appends a snapshot or step data to history.
        """
        self.history.append(step_data)
