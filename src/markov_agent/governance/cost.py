from pydantic import BaseModel, ConfigDict, Field

from markov_agent.engine.adk_wrapper import ADKConfig


class CostGovernor(BaseModel):
    """Governs model selection and budget tracking based on request complexity."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_budget: int = Field(default=1000000, description="Max allowed tokens.")
    cost_budget: float = Field(default=10.0, description="Max allowed cost in USD.")

    current_tokens: int = Field(default=0)
    current_cost: float = Field(default=0.0)

    # Pre-configured ADKConfigs for different tiers
    cheap_config: ADKConfig
    reasoning_config: ADKConfig
    standard_config: ADKConfig | None = Field(default=None)

    def route_request(self, complexity_score: float) -> ADKConfig:
        """Select an ADKConfig based on a complexity score [0.0 - 1.0]."""
        if complexity_score < 0.3:
            return self.cheap_config

        if complexity_score > 0.7:
            return self.reasoning_config

        return self.standard_config or self.cheap_config

    def check_budget(
        self, estimated_cost: float = 0.0, estimated_tokens: int = 0
    ) -> bool:
        """Check if the estimated execution would exceed the budget."""
        if (self.current_cost + estimated_cost) > self.cost_budget:
            return False

        return (self.current_tokens + estimated_tokens) <= self.token_budget

    def record_usage(self, cost: float = 0.0, tokens: int = 0) -> None:
        """Update the current consumption tracking."""
        self.current_cost += cost
        self.current_tokens += tokens
