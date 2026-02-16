import asyncio
import os
from typing import Literal

from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.simulation.scenario import Scenario, ScenarioManager
from markov_agent.simulation.twin import BaseDigitalTwin
from markov_agent.topology.graph import Graph

# ==============================================================================
# 1. State Definition (The "DNA")
# ==============================================================================


class CompanyState(BaseState):
    """The complete state of the company simulation."""

    # Resources
    budget: float = Field(default=1_000_000.0, description="Available capital in USD.")
    reputation: float = Field(default=100.0, description="Brand score (0-100).")

    # Project Status
    project_name: str = "Project Phoenix"
    features: list[str] = Field(default_factory=list)
    risk_level: Literal["Low", "Medium", "High", "Critical"] = "Medium"
    compliance_score: float = 100.0

    # Workflow Flags
    is_approved_by_finance: bool = False
    is_approved_by_legal: bool = False
    is_launched: bool = False

    # Audit Trail
    logs: list[str] = Field(
        default_factory=list, json_schema_extra={"behavior": "append"}
    )


# ==============================================================================
# 2. Structured Output Schemas
# ==============================================================================


class EngineeringOutput(BaseModel):
    cost: float = Field(description="Total estimated development cost.")
    reasoning: str = Field(description="Reasoning for the cost estimate.")


class FinanceOutput(BaseModel):
    is_approved: bool = Field(description="Whether the budget is approved.")
    reasoning: str = Field(description="CFO's reasoning.")


class LegalOutput(BaseModel):
    is_approved: bool = Field(description="Whether the project is legally compliant.")
    compliance_fix_applied: bool = Field(
        default=False, description="Whether legal fixes were applied."
    )
    reputation_hit: float = Field(
        default=0.0, description="Reputation loss due to privacy concerns."
    )
    reasoning: str = Field(description="General Counsel's reasoning.")


# ==============================================================================
# 3. Digital Twin (The "Shell" / Laws)
# ==============================================================================


class CompanyConstitution(BaseDigitalTwin[CompanyState]):
    """Enforces the immutable laws of the corporation."""

    async def validate_transition(
        self, current: CompanyState, proposed: CompanyState
    ) -> bool:
        # Law 1: Bankruptcy Protection
        if proposed.budget < 0:
            print(f"❌ [Twin] VETO: Budget cannot be negative ({proposed.budget})")
            return False

        # Law 2: Compliance Floor
        if proposed.is_launched and proposed.compliance_score < 50:
            print(
                f"❌ [Twin] VETO: Cannot launch illegal product (Compliance: {proposed.compliance_score})"
            )
            return False

        return True


# ==============================================================================
# 4. Department Logic (The "Nodes")
# ==============================================================================

# Common ADK Config
adk_config = ADKConfig(
    model_name="gemini-3-flash-preview", api_key=os.environ.get("GEMINI_API_KEY")
)


# Mock Responder for when API key is missing
def mock_company_responder(prompt: str) -> str:
    import json

    if "CTO" in prompt:
        cost = 100_000.0
        if "Surveillance" in prompt:
            cost = 500_000.0
        return json.dumps(
            {"cost": cost, "reasoning": "Estimated based on feature complexity."}
        )
    if "CFO" in prompt:
        is_approved = True
        # If budget is very low, reject
        if (
            "budget: 200000" in prompt.lower()
            or "budget: 600000" in prompt.lower()
            or "budget: 5000000" in prompt.lower()
        ):
            is_approved = True
        else:
            is_approved = False
        return json.dumps(
            {"is_approved": is_approved, "reasoning": "Budget looks acceptable."}
        )
    if "General Counsel" in prompt:
        is_approved = True
        return json.dumps(
            {
                "is_approved": is_approved,
                "compliance_fix_applied": False,
                "reputation_hit": 10.0 if "Surveillance" in prompt else 0.0,
                "reasoning": "Compliance review completed.",
            }
        )
    return "{}"


# --- Department: Product ---
def product_strategy(state: CompanyState) -> CompanyState:
    """Product Manager defines the scope."""
    s = state.update()

    if s.risk_level == "High" and "AI-Powered Surveillance" not in s.features:
        # We use a list for logs so it appends
        return s.update(
            features=[*s.features, "AI-Powered Surveillance"],
            compliance_score=s.compliance_score - 30,
            logs=["Product added 'AI Surveillance' feature."],
        )
    if "AI Chatbot" not in s.features:
        return s.update(
            features=[*s.features, "AI Chatbot"],
            logs=["Product added 'AI Chatbot' feature."],
        )

    return s


# --- LLM Prompts ---

engineering_prompt = """
You are the CTO. Analyze the requested features: {{state.features}}.
Estimate the development cost.
- 'AI-Powered Surveillance' costs $500,000.
- 'AI Chatbot' costs $100,000.

Return the total cost and your reasoning.
"""

finance_prompt = """
You are the CFO. Review the current budget: ${{state.budget}} and risk level: {{state.risk_level}}.
Decision Policy:
- If Budget < $100,000, REJECT.
- If Risk is 'Critical', REJECT.
- Otherwise, APPROVE.

Return your decision and reasoning.
"""

legal_prompt = """
You are General Counsel. Review 'compliance_score' ({{state.compliance_score}}) and 'features'.
Policy:
- If compliance_score < 70, require changes.
- If features contains "Surveillance", decrease 'reputation' by 10.
- If compliance_score >= 70, approve.

Return your decision and any adjustments.
"""

# ==============================================================================
# 5. Topology Setup
# ==============================================================================


def build_company_graph() -> Graph:
    graph = Graph(
        name="CorpSim_v1", state_type=CompanyState, twin=CompanyConstitution()
    )

    # Register Nodes
    graph.task(product_strategy, name="ProductStrategy")

    mock = None if os.environ.get("GEMINI_API_KEY") else mock_company_responder

    @graph.node(
        adk_config=adk_config,
        name="EngineeringDept",
        output_schema=EngineeringOutput,
        mock_responder=mock,
    )
    def engineering_dept(state: CompanyState, result: EngineeringOutput):
        return state.update(
            budget=state.budget - result.cost,
            logs=[f"CTO: {result.reasoning} (Cost: ${result.cost:,.2f})"],
        )

    engineering_dept.__doc__ = engineering_prompt

    @graph.node(
        adk_config=adk_config,
        name="FinanceDept",
        output_schema=FinanceOutput,
        mock_responder=mock,
    )
    def finance_dept(state: CompanyState, result: FinanceOutput):
        return state.update(
            is_approved_by_finance=result.is_approved,
            logs=[f"CFO: {result.reasoning} (Approved: {result.is_approved})"],
        )

    finance_dept.__doc__ = finance_prompt

    @graph.node(
        adk_config=adk_config,
        name="LegalDept",
        output_schema=LegalOutput,
        mock_responder=mock,
    )
    def legal_dept(state: CompanyState, result: LegalOutput):
        new_compliance = state.compliance_score
        if result.compliance_fix_applied:
            new_compliance += 10

        return state.update(
            is_approved_by_legal=result.is_approved,
            reputation=state.reputation - result.reputation_hit,
            compliance_score=new_compliance,
            logs=[f"Legal: {result.reasoning} (Approved: {result.is_approved})"],
        )

    legal_dept.__doc__ = legal_prompt

    @graph.task
    def launch_day(state: CompanyState):
        s = state.update(is_launched=True, logs=["🚀 PRODUCT LAUNCHED!"])
        s.record_reward(100.0)
        return s

    @graph.task
    def bankruptcy(state: CompanyState):
        s = state.update(logs=["💀 COMPANY BANKRUPT."])
        s.record_reward(-100.0)
        return s

    @graph.task
    def market_simulation(state: CompanyState):
        if state.reputation > 80:
            s = state.update(
                budget=state.budget + 500_000, logs=["Market loves it! Revenue +$500k"]
            )
            s.record_reward(50.0)
            return s
        return state.update(logs=["Market backlash. Revenue stagnant."])

    @graph.task(name="CEO_Office")
    def ceo_node(state: CompanyState):
        return state

    graph.add_transition("ProductStrategy", "EngineeringDept")
    graph.add_transition("EngineeringDept", "FinanceDept")
    graph.add_transition("FinanceDept", "CEO_Office")
    graph.add_transition("LegalDept", "CEO_Office")

    graph.route(
        "CEO_Office",
        {
            "ProductStrategy": lambda s: (
                not s.is_approved_by_finance and s.budget >= 50_000
            ),
            "Bankruptcy": lambda s: not s.is_approved_by_finance and s.budget < 50_000,
            "LegalDept": lambda s: (
                s.is_approved_by_finance and not s.is_approved_by_legal
            ),
            "LaunchDay": lambda s: s.is_approved_by_finance and s.is_approved_by_legal,
        },
    )

    graph.add_transition("LaunchDay", "MarketSimulation")
    graph.entry_point = "ProductStrategy"

    return graph


# ==============================================================================
# 6. Execution & Analysis
# ==============================================================================


async def main():
    print("\n🏢 Initializing Company Digital Twin...")
    graph = build_company_graph()

    manager = ScenarioManager(graph)
    base_state = CompanyState()

    scenarios = [
        Scenario(
            name="Safe Mode",
            state_overrides={"risk_level": "Low", "budget": 200_000.0},
            n_runs=1,
        ),
        Scenario(
            name="Crisis Mode",
            state_overrides={"risk_level": "High", "budget": 600_000.0},
            n_runs=1,
        ),
        Scenario(
            name="Deep Pockets",
            state_overrides={"risk_level": "High", "budget": 5_000_000.0},
            n_runs=1,
        ),
    ]

    print("🧠 Running Strategic Analysis (Monte Carlo)...")
    results = await manager.run_scenarios(base_state, scenarios, max_concurrency=2)

    print("\n📊 --- ANALYSIS REPORT ---")
    for res in results:
        print(f"\nScenario: {res.scenario_name}")
        print(f"  Success Rate (Launch): {res.success_rate:.1%}")
        print(f"  Avg Reward: {res.avg_reward:.2f}")
        print(f"  Avg Steps: {res.avg_steps:.1f}")

        if res.runs:
            example = res.runs[0]
            print("  Trace Excerpt:")
            for log in example.final_state.logs[-3:]:
                print(f"    - {log}")


if __name__ == "__main__":
    asyncio.run(main())
