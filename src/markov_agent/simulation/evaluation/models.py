from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """
    Represents a single evaluation scenario.
    """

    id: str = Field(..., description="Unique identifier for this test case.")
    description: str = Field(..., description="What this case tests.")
    user_persona: str = Field(
        ...,
        description="Description of the simulated user (e.g., 'Impatient traveler').",
    )
    user_goal: str = Field(
        ..., description="The specific objective the simulated user wants to achieve."
    )
    expected_outcome: str | None = Field(
        None, description="Description of the ideal result."
    )


class TurnResult(BaseModel):
    """
    Captures a single exchange in the conversation.
    """

    user_input: str
    agent_response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """
    Metrics calculated for a session.
    """

    scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class SessionResult(BaseModel):
    """
    The outcome of running one EvalCase.
    """

    case_id: str
    success: bool
    history: list[TurnResult] = Field(default_factory=list)
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    error: str | None = None
