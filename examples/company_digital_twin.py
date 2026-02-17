import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from markov_agent import ADKConfig, BaseDigitalTwin, ResourceGovernor
from markov_agent.containers.swarm import Swarm
from markov_agent.core.state import BaseState
from markov_agent.engine.ppu import ProbabilisticNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. State Definition (Complex Corporate Metrics)
# ==============================================================================


class CompanyState(BaseState):
    """Global state for the complex corporate swarm."""

    # Financials (in USD)
    budget: float = Field(default=2_000_000.0)
    revenue: float = Field(default=0.0)
    burn_rate: float = Field(default=100_000.0)

    # Market & Product (0-100 scale where appropriate)
    market_share: float = Field(default=5.0)  # Percentage
    reputation: float = Field(default=70.0)
    product_quality: float = Field(default=60.0)
    technical_debt: float = Field(default=20.0)
    innovation_index: float = Field(default=30.0)
    market_sentiment: float = Field(default=50.0)

    # Human Capital
    talent_index: float = Field(default=50.0)
    employee_happiness: float = Field(default=75.0)

    # External
    competitor_aggression: float = Field(default=20.0)
    regulatory_risk: float = Field(default=10.0)

    # Control Flow
    next_action: str = "CEO"
    quarter: int = 1
    max_quarters: int = 4
    is_bankrupt: bool = False
    is_finished: bool = False

    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append"}
    )

    def record_log(self, message: str) -> "CompanyState":
        """Helper to keep logs manageable by keeping only recent ones."""
        new_logs = self.logs + [message]
        if len(new_logs) > 50:
            new_logs = new_logs[-50:]
        return self.update(logs=new_logs)


# ==============================================================================
# 3. LLM Configuration (Local LLM via LiteLLM)
# ==============================================================================

adk_config = ADKConfig(
    model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
    api_base="http://192.168.1.213:8080/v1",
    api_key="no-key",
    use_litellm=True,
    temperature=0.7,
    max_input_tokens=9000,
    reduction_prompt=(
        "You are a Corporate Secretary. Summarize the following company state, focus on "
        "KPIs (Budget, Revenue, Market Share) and key department achievements. "
        "Keep the summary strictly professional and under the token limit."
    ),
)

# ==============================================================================
# 4. Output Schemas
# ==============================================================================


class CEOOutput(BaseModel):
    next_action: Literal[
        "Marketing", "Engineering", "HR", "Operations", "RD", "Legal", "EndSimulation"
    ]
    strategy: str
    rationale: str
    budget_allocation: float = Field(
        description="Total budget allocated for the next move."
    )


class DeptOutput(BaseModel):
    cost: float
    report: str
    metrics: dict[str, float]
    risks_identified: list[str] = Field(default_factory=list)


# ==============================================================================
# 5. World Model & Digital Twin (Physics of Business)
# ==============================================================================


class BusinessTwin(BaseDigitalTwin[CompanyState]):
    """Simulates market dynamics and enforces physical business constraints."""

    async def validate_transition(
        self, current: CompanyState, proposed: CompanyState
    ) -> bool:
        # Check for catastrophic failure (Debt limit)
        if proposed.budget < -1_000_000:
            print("🛑 [Twin] VETO: Debt limit reached. Proposed action blocked.")
            return False
        return True

    async def evolve_world(self, state: CompanyState) -> CompanyState:
        """Simulate revenue generation and decay between turns."""
        # Normalize metrics to prevent runaway values
        def clamp(val: float, min_v: float = 0.0, max_v: float = 100.0) -> float:
            return max(min_v, min(max_v, val))

        # 1. Market Growth Multipliers (Normalized)
        quality_score = clamp(state.product_quality) / 100.0
        reputation_score = clamp(state.reputation) / 100.0
        sentiment_score = clamp(state.market_sentiment) / 100.0

        # 2. Revenue Calculation (Balanced)
        # Revenue = (Market Share * Total Market Size) * (Quality * Reputation * Sentiment)
        total_market_value = 5_000_000.0
        market_capture = (state.market_share / 100.0) * total_market_value
        performance_multiplier = 0.5 + (
            (quality_score + reputation_score + sentiment_score) / 3.0
        )
        generated_revenue = market_capture * performance_multiplier

        # 3. Burn Rate Adjustments (Tech Debt increases burn)
        debt_penalty = (clamp(state.technical_debt) / 100.0) * 0.5
        actual_burn = state.burn_rate * (1 + debt_penalty)

        # 4. Financial Update
        new_budget = state.budget + generated_revenue - actual_burn

        # 5. Natural Entropy (Things get worse if not maintained)
        new_reputation = clamp(state.reputation - 2.0)
        new_quality = clamp(state.product_quality - 1.0)
        new_debt = clamp(state.technical_debt + 1.5)
        new_sentiment = clamp(state.market_sentiment + (state.innovation_index / 100.0) - 2.0)

        return state.update(
            budget=new_budget,
            revenue=state.revenue + generated_revenue,
            reputation=new_reputation,
            product_quality=new_quality,
            technical_debt=new_debt,
            market_sentiment=new_sentiment,
            logs=[
                f"Market Update: Generated ${generated_revenue:,.0f} revenue with {state.market_share:.1f}% share."
            ],
        )


# ==============================================================================
# 6. Topology Construction
# ==============================================================================


def build_complex_corp() -> Swarm:
    governor = ResourceGovernor(memory_threshold_percent=85.0)

    # Helper to handle state updates from either dict or CompanyState
    def _safe_update(state: Any, **kwargs: Any) -> Any:
        if hasattr(state, "update"):
            return state.update(**kwargs)
        # If it's a dict (fallback for some ADK runs)
        new_state = state.copy()
        new_state.update(kwargs)
        return new_state

    def _safe_get(state: Any, key: str) -> Any:
        if hasattr(state, key):
            return getattr(state, key)
        if isinstance(state, dict):
            return state.get(key)
        return None

    # 2. CEO Node
    ceo = ProbabilisticNode(
        name="CEO",
        adk_config=adk_config,
        output_schema=CEOOutput,
        prompt_template="""CEO STRATEGIC DASHBOARD
        Quarter: {{quarter}} | Budget: ${{budget:,.0f}} | Total Revenue: ${{revenue:,.0f}}
        
        KPI METRICS (0-100 scale):
        - Market Share: {{market_share}}%
        - Quality: {{product_quality}}
        - Innovation: {{innovation_index}}
        - Tech Debt: {{technical_debt}}
        - Sentiment: {{market_sentiment}}
        
        LOGS:
        {% for log in logs[-5:] %}
        - {{ log }}
        {% endfor %}

        As CEO, select the next department to activate. 
        RD boosts Innovation. Marketing boosts Share/Sentiment. Engineering boosts Quality/reduces Debt.
        Ops reduces Burn. Legal reduces Risk. HR improves Talent.
        """,
        state_updater=lambda state, result: _safe_update(
            state,
            next_action=result.next_action,
            logs=[
                f"CEO Strategy: {result.strategy} -> {result.next_action} (Allocated: ${result.budget_allocation:,.0f})"
            ],
        ),
    )

    # 3. Department Factory
    def create_dept(name: str, goal: str) -> ProbabilisticNode:
        return ProbabilisticNode(
            name=name,
            adk_config=adk_config,
            output_schema=DeptOutput,
            prompt_template=f"""DEPARTMENT EXECUTION: {name}
            GOAL: {goal}
            
            CORPORATE STATE:
            {{{{state.model_dump_json()}}}}

            Execute your mission. Specify the cost and impact on metrics (use small increments like 1-10).
            """,
            state_updater=lambda state, result: _safe_update(
                state,
                **{
                    "budget": _safe_get(state, "budget") - result.cost,
                    "next_action": "CEO",
                    "logs": [f"{name} Report: {result.report} (Cost: ${result.cost:,.0f})"],
                    **{
                        k: _safe_get(state, k) + v
                        for k, v in result.metrics.items()
                        if _safe_get(state, k) is not None
                    },
                },
            ),
        )

    marketing = create_dept(
        "Marketing", "Boost brand reputation and capture market share."
    )
    engineering = create_dept(
        "Engineering", "Improve core quality and pay down technical debt."
    )
    hr = create_dept("HR", "Optimize talent acquisition and employee happiness.")
    ops = create_dept("Operations", "Efficiency improvements to lower burn rate.")
    rd = create_dept("RD", "Research and development for innovation breakthroughs.")
    legal = create_dept("Legal", "Risk mitigation and regulatory compliance.")

    return Swarm(
        name="ComplexCorp",
        supervisor=ceo,
        workers=[marketing, engineering, hr, ops, rd, legal],
        router_func=lambda s: (
            (s.next_action if s.next_action != "EndSimulation" else None)
            if hasattr(s, "next_action")
            else (s["next_action"] if s["next_action"] != "EndSimulation" else None)
        ),
        state_type=CompanyState,
        governor=governor,
        twin=BusinessTwin(),
    )


# ==============================================================================
# 7. Main Execution
# ==============================================================================


async def main():
    print("🏢 Starting Balanced Complex Corporate Simulation...")
    swarm = build_complex_corp()
    twin = BusinessTwin()

    state = CompanyState(
        budget=2_000_000.0,
        logs=["Simulation Start: Capital $2.0M"],
    )

    for q in range(1, 5):
        print(f"\n[Quarter {q}]")
        state = state.update(quarter=q)

        for step in range(3):
            try:
                state = await swarm.run(state)
                if state.next_action == "EndSimulation":
                    break
            except Exception as e:
                logger.error("Step %s execution error: %s", step + 1, e)
                state = state.update(next_action="CEO")

        # Evolve world at end of quarter
        state = await twin.evolve_world(state)

        print(f"  > Budget: ${state.budget:,.0f} | Share: {state.market_share:.1f}%")
        print(
            f"  > Innovation: {state.innovation_index:.1f} | Debt: {state.technical_debt:.1f}"
        )

        if state.budget < -1_000_000:
            print("💀 INSOLVENCY. The company has collapsed.")
            break

    print("\n📊 --- FINAL PERFORMANCE REPORT ---")
    print(f"Final Budget: ${state.budget:,.2f}")
    print(f"Total Revenue: ${state.revenue:,.2f}")
    print(f"Final Innovation: {state.innovation_index:.1f}")
    print(f"Final Sentiment: {state.market_sentiment:.1f}")
    print("Simulation Complete.")


if __name__ == "__main__":
    asyncio.run(main())
