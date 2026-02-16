import asyncio
import json
import os
import random
from typing import Literal

from pydantic import BaseModel, Field

from markov_agent import ADKConfig, BaseDigitalTwin, ResourceGovernor, setup_llm_logging
from markov_agent.containers.swarm import Swarm
from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph

# ==============================================================================
# 1. Setup Logging (Save all LLM logs to ./xxx.log)
# ==============================================================================
setup_llm_logging(log_file="./xxx.log")

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

    # Human Capital
    talent_index: float = Field(default=50.0)  # 0-100
    employee_happiness: float = Field(default=75.0)

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
# 3. Local LLM Configuration
# ==============================================================================

local_adk_config = ADKConfig(
    model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
    api_base=os.environ.get("LOCAL_LLM_API_BASE", "http://192.168.1.213:8080/v1"),
    api_key="no-key",
    use_litellm=True,
    temperature=0.7,
)


def complex_local_llm_mock(prompt: str) -> str:
    lp = prompt.lower()
    if "ceo" in lp or "strategy" in lp:
        # Complex logic for CEO mock
        choices = ["Marketing", "Engineering", "HR", "Operations"]
        next_dept = random.choice(choices)
        if "quarter 4" in lp:
            next_dept = "EndSimulation"
        return json.dumps(
            {
                "next_action": next_dept,
                "strategy": f"Focusing on {next_dept} to stabilize the company.",
            }
        )

    if "marketing" in lp:
        return json.dumps(
            {
                "cost": 250000.0,
                "report": "Omnichannel campaign 'Phoenix' launched successfully.",
                "metrics": {"reputation": 8.5, "market_share": 1.2},
            }
        )

    if "engineering" in lp:
        return json.dumps(
            {
                "cost": 400000.0,
                "report": "Refactored core engine and added 3 major features.",
                "metrics": {"product_quality": 12.0, "technical_debt": -5.0},
            }
        )

    if "hr" in lp:
        return json.dumps(
            {
                "cost": 150000.0,
                "report": "New equity package and remote work policy implemented.",
                "metrics": {"talent_index": 10.0, "employee_happiness": 5.0},
            }
        )

    if "operations" in lp:
        return json.dumps(
            {
                "cost": 100000.0,
                "report": "Optimized cloud costs and vendor contracts.",
                "metrics": {"burn_rate": -15000.0},
            }
        )

    # Generic response that fits DeptOutput if everything else fails
    return json.dumps(
        {
            "cost": 50000.0,
            "report": "Standard operational task completed.",
            "metrics": {},
        }
    )


mock_provider = (
    None if os.environ.get("USE_REAL_LLM") == "true" else complex_local_llm_mock
)

# ==============================================================================
# 4. Output Schemas
# ==============================================================================


class CEOOutput(BaseModel):
    next_action: Literal[
        "Marketing", "Engineering", "HR", "Operations", "EndSimulation"
    ]
    strategy: str


class DeptOutput(BaseModel):
    cost: float
    report: str
    metrics: dict[str, float]


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
        # Revenue depends on (Market Share * Reputation * Product Quality)
        base_revenue = state.market_share * 200_000
        quality_multiplier = state.product_quality / 50
        reputation_multiplier = state.reputation / 70
        talent_multiplier = state.talent_index / 50

        generated_revenue = (
            base_revenue
            * quality_multiplier
            * reputation_multiplier
            * talent_multiplier
        )

        # Apply burn rate and taxes (15%)
        net_revenue = generated_revenue * 0.85
        new_budget = state.budget + net_revenue - state.burn_rate

        # Natural Decay
        new_reputation = max(0, state.reputation - 2)  # Market forgetfulness
        new_debt = state.technical_debt + 1  # Entropy

        return state.update(
            budget=new_budget,
            revenue=state.revenue + generated_revenue,
            reputation=new_reputation,
            technical_debt=new_debt,
            logs=[f"Market: Generated ${generated_revenue:,.0f} revenue."],
        )


# ==============================================================================
# 6. Topology Construction
# ==============================================================================


def build_complex_corp() -> Swarm:
    # 1. Resource Protection
    governor = ResourceGovernor(memory_threshold_percent=85.0)

    # 2. CEO Node
    ceo = Graph(name="CEO", default_adk_config=local_adk_config)

    @ceo.node(name="Strategy", output_schema=CEOOutput, mock_responder=mock_provider)
    def ceo_strategy(state: CompanyState, result: CEOOutput):
        """CEO Strategic Decision Node.
        Current Quarter: {{state.quarter}}
        Budget: {{state.budget}}
        Metrics: Share={{state.market_share}}%, Quality={{state.product_quality}}%
        """
        return state.update(
            next_action=result.next_action,
            logs=[f"CEO: {result.strategy}"],
        )

    ceo.entry_point = "Strategy"

    # 3. Department Factory
    def create_dept(name: str, task: str) -> Graph:
        g = Graph(name=name, default_adk_config=local_adk_config)

        @g.node(name="Execute", output_schema=DeptOutput, mock_responder=mock_provider)
        def exec_dept(state: CompanyState, result: DeptOutput):
            """Execute department task.
            Budget: {{state.budget}}
            """
            updates = {
                "budget": state.budget - result.cost,
                "next_action": "CEO",
                "logs": [f"{name}: {result.report}"],
            }
            # Merge metrics
            for k, v in result.metrics.items():
                if hasattr(state, k):
                    curr = getattr(state, k)
                    updates[k] = curr + v

            return state.update(**updates)

        g.entry_point = "Execute"
        return g

    marketing = create_dept("Marketing", "Boost reputation and market share.")
    engineering = create_dept("Engineering", "Improve quality and reduce debt.")
    hr = create_dept("HR", "Hire talent and improve happiness.")
    ops = create_dept("Operations", "Reduce burn rate.")

    # 4. Assembly
    return Swarm(
        name="ComplexCorp",
        supervisor=ceo.as_node(),
        workers=[
            marketing.as_node(),
            engineering.as_node(),
            hr.as_node(),
            ops.as_node(),
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
        budget=1_500_000.0,
        logs=["Simulation Start: Seed capital $1.5M"],
    )

    # Simulation runs for 4 Quarters
    for q in range(1, 5):
        print(f"\n--- Quarter {q} ---")
        state = state.update(quarter=q)

        # Run Swarm for 3 steps per quarter (multiple decisions)
        for _step in range(3):
            try:
                state = await swarm.run(state)
                if state.next_action == "EndSimulation":
                    break
            except MemoryError as e:
                print(f"🚨 RESOURCE HALT: {e}")
                return

        # End of Quarter Market Evolution
        state = await twin.evolve_world(state)

        print(f"Budget: ${state.budget:,.0f} | Revenue: ${state.revenue:,.0f}")
        print(
            f"Share: {state.market_share:.1f}% | Quality: {state.product_quality:.1f}"
        )

        if state.budget < 0:
            print("💸 BANKRUPTCY. Company dissolved.")
            break

    print("\n📊 --- Final Results ---")
    print(f"Total Revenue: ${state.revenue:,.2f}")
    print(f"Final Market Share: {state.market_share:.2f}%")
    print(f"Final Talent Index: {state.talent_index:.1f}")

    print("\n✅ Check ./xxx.log for detailed LLM logs.")


if __name__ == "__main__":
    asyncio.run(main())
