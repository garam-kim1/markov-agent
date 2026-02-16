from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState


@runtime_checkable
class DigitalTwin[StateT: BaseState](Protocol):
    """An Active Digital Twin that enforces physical laws and business logic.

    In the Adaptive Systems paradigm, the Twin is the 'Shell'—the immutable
    layer that maintains identity and enforces constraints.
    """

    async def validate_transition(self, current: StateT, proposed: StateT) -> bool:
        """Check if the proposed transition violates any 'physical' or business laws."""
        ...

    async def calculate_consequences(self, state: StateT, action: Any) -> StateT:
        """Simulate the next state based on an action, acting as a world model."""
        ...


class BaseDigitalTwin[StateT: BaseState](BaseModel):
    """Base implementation for a Digital Twin."""

    constraints: dict[str, Any] = Field(default_factory=dict)

    async def validate_transition(self, current: StateT, proposed: StateT) -> bool:
        """Default validation: allow everything unless overridden."""
        return True

    async def calculate_consequences(self, state: StateT, action: Any) -> StateT:
        """Default simulation: just return the state (no-op)."""
        return state


class WorldModel[StateT: BaseState]:
    """A wrapper that allows an Agent to 'Dream' using its Digital Twin.

    This implements the 'Offline Simulation' phase described in the report.
    """

    def __init__(self, twin: DigitalTwin[StateT]):
        self.twin = twin

    async def predict(self, state: StateT, action: Any) -> StateT:
        """Predict the outcome of an action without executing it in the real world."""
        return await self.twin.calculate_consequences(state, action)

    async def evaluate_safety(self, current: StateT, proposed: StateT) -> float:
        """Return a safety score [0, 1] for a proposed transition."""
        is_valid = await self.twin.validate_transition(current, proposed)
        return 1.0 if is_valid else 0.0
