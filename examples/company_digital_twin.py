import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from markov_agent import ADKConfig, BaseDigitalTwin, ResourceGovernor
from markov_agent.containers.swarm import Swarm
from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph

# ==============================================================================
# 2. State Definition (Complex Corporate Metrics)
# ==============================================================================


class CompanyState(BaseState):
    """Global state for the complex corporate swarm."""

    # Financials
    budget: float = Field(default=2_000_000.0)
    revenue: float = Field(default=0.0)
    burn_rate: float = Field(default=100_000.0)  # Fixed quarterly overhead

    # Market & Product
    market_share: float = Field(default=2.0)
    reputation: float = Field(default=70.0)
    product_quality: float = Field(default=60.0)
    technical_debt: float = Field(default=20.0)
    innovation_index: float = Field(default=30.0)

    # Human Capital
    talent_index: float = Field(default=50.0)  # 0-100
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

    department_reports: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append"}
    )


# ==============================================================================
# 3. LLM Configuration (Local LLM via LiteLLM)
# ==============================================================================

adk_config = ADKConfig(
    model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
    api_base="http://192.168.1.213:8080/v1",
    api_key="no-key",
    use_litellm=True,
    temperature=0.7,
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


class DeptOutput(BaseModel):
    cost: float
    report: str
    metrics: dict[str, float]
    risks_identified: list[str] = Field(default_factory=list)


# ==============================================================================
# 5. World Model & Digital Twin (Physics of Business)
# ==============================================================================


class BusinessTwin(BaseDigitalTwin[CompanyState]):
    """Simulates market dynamics and enforces constraints."""

    async def validate_transition(
        self, current: CompanyState, proposed: CompanyState
    ) -> bool:
        # Check for bankruptcy
        if proposed.budget < 0:
            print("🛑 [Twin] VETO: Immediate bankruptcy prevented. Reverting.")
            return False
        return True

    async def evolve_world(self, state: CompanyState) -> CompanyState:
        """Simulate revenue generation and decay between turns."""
        # Revenue depends on (Market Share * Reputation * Product Quality * Innovation)
        base_revenue = state.market_share * 200_000
        quality_multiplier = state.product_quality / 50
        reputation_multiplier = state.reputation / 70
        talent_multiplier = state.talent_index / 50
        innovation_multiplier = state.innovation_index / 30

        generated_revenue = (
            base_revenue
            * quality_multiplier
            * reputation_multiplier
            * talent_multiplier
            * innovation_multiplier
        )

        # Apply burn rate and taxes (15%)
        net_revenue = generated_revenue * 0.85
        new_budget = state.budget + net_revenue - state.burn_rate

        # Natural Decay & External Pressures
        new_reputation = max(
            0, state.reputation - (2 + state.competitor_aggression / 20)
        )
        new_debt = state.technical_debt + 1.5  # Entropy
        new_innovation = max(0, state.innovation_index - 1)

        return state.update(
            budget=new_budget,
            revenue=state.revenue + generated_revenue,
            reputation=new_reputation,
            technical_debt=new_debt,
            innovation_index=new_innovation,
            logs=[f"Market: Generated ${generated_revenue:,.0f} revenue."],
        )


# ==============================================================================
# 6. Topology Construction
# ==============================================================================


def build_complex_corp() -> Swarm:
    # 1. Resource Protection
    governor = ResourceGovernor(memory_threshold_percent=85.0)

    # 2. CEO Node
    ceo = Graph(name="CEO", default_adk_config=adk_config)

    @ceo.node(name="Strategy", output_schema=CEOOutput)
    def ceo_strategy(state: CompanyState, result: CEOOutput):
        """CEO Strategic Decision Node.
        Current Quarter: {{state.quarter}}
        Budget: {{state.budget}}
        Metrics: Share={{state.market_share}}%, Quality={{state.product_quality}}%, Innovation={{state.innovation_index}}
        External: Aggression={{state.competitor_aggression}}, Risk={{state.regulatory_risk}}

        Decide which department needs investment to ensure long-term sustainability and growth.
        """
        return state.update(
            next_action=result.next_action,
            logs=[f"CEO Strategy: {result.strategy} (Rationale: {result.rationale})"],
        )

    ceo.entry_point = "Strategy"

    # 3. Department Factory
    def create_dept(name: str, goal: str) -> Graph:
        g = Graph(name=name, default_adk_config=adk_config)

        @g.node(name="Execute", output_schema=DeptOutput)
        def exec_dept(state: CompanyState, result: DeptOutput):
            """Execute department task for {{name}}.
            Goal: {{goal}}
            Current Budget: {{state.budget}}
            Current Metrics: {{state.model_dump_json()}}

            Provide a report on actions taken, costs incurred, and impact on metrics.
            """
            updates = {
                "budget": state.budget - result.cost,
                "next_action": "CEO",
                "logs": [f"{name} Report: {result.report} (Cost: ${result.cost:,.0f})"],
            }
            # Merge metrics
            for k, v in result.metrics.items():
                if hasattr(state, k):
                    curr = getattr(state, k)
                    updates[k] = curr + v

            if result.risks_identified:
                updates["logs"].append(
                    f"{name} Risks: {', '.join(result.risks_identified)}"
                )

            return state.update(**updates)

        g.entry_point = "Execute"
        return g

    marketing = create_dept("Marketing", "Boost reputation and market share.")
    engineering = create_dept(
        "Engineering", "Improve quality and reduce technical debt."
    )
    hr = create_dept("HR", "Hire talent and improve employee happiness.")
    ops = create_dept("Operations", "Reduce burn rate and optimize efficiency.")
    rd = create_dept("RD", "Increase innovation index and develop new IP.")
    legal = create_dept("Legal", "Decrease regulatory risk and handle compliance.")

    # 4. Assembly
    return Swarm(
        name="ComplexCorp",
        supervisor=ceo.as_node(),
        workers=[
            marketing.as_node(),
            engineering.as_node(),
            hr.as_node(),
            ops.as_node(),
            rd.as_node(),
            legal.as_node(),
        ],
        router_func=lambda s: (
            s.next_action if s.next_action != "EndSimulation" else None
        ),
        state_type=CompanyState,
        governor=governor,
        twin=BusinessTwin(),
    )


# ==============================================================================
# 7. Main Loop
# ==============================================================================


async def main():
    print("🏢 Starting Complex Corporate Simulation...")
    swarm = build_complex_corp()
    twin = BusinessTwin()

    state = CompanyState(
        budget=2_000_000.0,
        logs=["Simulation Start: Seed capital $2M"],
    )

    # Simulation runs for 4 Quarters
    for q in range(1, 5):
        print(f"\n--- Quarter {q} ---")
        state = state.update(quarter=q)

        # Run Swarm for 5 steps per quarter (more complex decisions)
        for _step in range(5):
            try:
                state = await swarm.run(state)
                if state.next_action == "EndSimulation":
                    break
            except Exception as e:
                print(f"🚨 EXECUTION ERROR: {e}")
                # We don't want to stop the whole simulation on a single node failure
                # but in a real case we might want to retry or handle it.
                state = state.update(next_action="CEO")
                continue

        # End of Quarter Market Evolution
        state = await twin.evolve_world(state)

        print(f"Budget: ${state.budget:,.0f} | Revenue: ${state.revenue:,.0f}")
        print(
            f"Share: {state.market_share:.1f}% | Quality: {state.product_quality:.1f} | Innovation: {state.innovation_index:.1f}"
        )

        if state.budget < 0:
            print("💸 BANKRUPTCY. Company dissolved.")
            break

    print("\n📊 --- Final Results ---")
    print(f"Total Revenue: ${state.revenue:,.2f}")
    print(f"Final Market Share: {state.market_share:.2f}%")
    print(f"Final Talent Index: {state.talent_index:.1f}")
    print(f"Final Innovation: {state.innovation_index:.1f}")

    print("\n✅ Simulation Complete.")


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
