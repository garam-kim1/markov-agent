import asyncio
import logging
import os
import random
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from markov_agent import ADKConfig, BaseState, Graph, ProbabilisticNode

# Configure logging with Rich
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("company_sim")
console = Console()

# ==============================================================================
# 1. Configuration
# ==============================================================================

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"

adk_config = ADKConfig(
    model_name=MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7,
    max_input_tokens=9000,
    use_litellm=True if not API_KEY else False,
)

if not API_KEY:
    console.print(
        "[yellow]No GEMINI_API_KEY found. Attempting to use local/default config.[/yellow]"
    )
    adk_config = ADKConfig(
        model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
        api_base="http://192.168.1.213:8080/v1",
        api_key="no-key",
        use_litellm=True,
        temperature=0.7,
        max_input_tokens=9000,
    )

# ==============================================================================
# 2. State Definition (Complex Corporate Metrics)
# ==============================================================================


class CompanyState(BaseState):
    """Global state for the complex corporate simulation."""

    # Financials (in USD)
    budget: float = Field(default=2_000_000.0)
    revenue: float = Field(default=0.0)
    burn_rate: float = Field(default=100_000.0)

    # Market & Product (0-100 scale)
    market_share: float = Field(default=5.0)
    reputation: float = Field(default=70.0)
    product_quality: float = Field(default=60.0)
    technical_debt: float = Field(default=20.0)
    innovation_index: float = Field(default=30.0)
    market_sentiment: float = Field(default=50.0)

    # Internal
    employee_happiness: float = Field(default=75.0)

    # Simulation Control
    turn: int = 1
    quarter: int = 1
    last_event: str = "Market is stable."
    next_action: str = "CEO"
    logs: list[str] = Field(default_factory=list)

    # CEO Proposals (System 1)
    proposed_department: str = ""
    proposed_budget: float = 0.0
    ceo_instruction: str = ""

    # Board Feedback (System 2)
    board_warning: str = ""

    def log(self, message: str) -> "CompanyState":
        """Add a log entry."""
        new_logs = [*self.logs, message]
        if len(new_logs) > 20:
            new_logs = new_logs[-20:]
        return self.update(logs=new_logs)


# ==============================================================================
# 3. Output Schemas
# ==============================================================================


class CEOOutput(BaseModel):
    analysis: str = Field(description="Brief analysis of current situation and risks.")
    department: Literal["Marketing", "Engineering", "HR", "Operations", "RD", "Legal"]
    instruction: str = Field(description="Specific instruction for the department.")
    budget_allocation: float = Field(description="Budget to allocate (max $500k).")


class BoardOutput(BaseModel):
    risk_assessment: str = Field(description="Evaluation of the CEO's proposal.")
    approved: bool = Field(
        description="True to approve the CEO's plan, False to reject."
    )
    feedback_for_ceo: str = Field(
        description="Feedback or constraints to give the CEO if rejected or approved."
    )


class DeptOutput(BaseModel):
    plan: str = Field(description="The action plan.")
    expected_cost: float
    focus_area: str = Field(description="Primary metric to improve.")


# ==============================================================================
# 4. Logic & Simulation (The Physics)
# ==============================================================================


def generate_market_event() -> tuple[str, dict[str, float]]:
    events = [
        ("Tech Boom", {"market_sentiment": 10.0}),
        ("Market Crash", {"market_sentiment": -15.0, "budget": -50000.0}),
        ("Viral Competitor Product", {"market_share": -2.0, "reputation": -5.0}),
        ("Regulatory Crackdown", {"technical_debt": 5.0, "budget": -20000.0}),
        ("Talent Shortage", {"employee_happiness": -10.0, "burn_rate": 20000.0}),
        ("Stable Market", {"market_sentiment": 1.0}),
    ]
    weights = [0.1, 0.05, 0.1, 0.05, 0.1, 0.6]
    event, impacts = random.choices(events, weights=weights, k=1)[0]
    return event, impacts


def simulate_department_action(
    state: CompanyState, dept_name: str, output: DeptOutput
) -> CompanyState:
    """Physics Engine: deterministically calculate the result of an action."""

    success_chance = 0.7 + (state.employee_happiness / 500.0)
    if state.budget < 0:
        success_chance -= 0.3

    is_success = random.random() < success_chance

    # Cap cost at both the allocated budget and the total remaining company budget
    actual_cost = min(output.expected_cost, state.proposed_budget, state.budget)
    msg = ""
    if actual_cost < output.expected_cost:
        msg = f" (Budget constrained: req ${output.expected_cost / 1000:.0f}k -> got ${actual_cost / 1000:.0f}k)"

    impacts = {"budget": -actual_cost}
    efficiency = (actual_cost / 100_000.0) * (1.5 if is_success else 0.5)

    if dept_name == "Marketing":
        impacts["market_share"] = 1.5 * efficiency
        impacts["market_sentiment"] = 2.0 * efficiency
        impacts["reputation"] = 1.0 * efficiency
    elif dept_name == "Engineering":
        impacts["product_quality"] = 2.0 * efficiency
        impacts["technical_debt"] = -3.0 * efficiency
        impacts["burn_rate"] = -5000 * efficiency
    elif dept_name == "HR":
        impacts["employee_happiness"] = 5.0 * efficiency
        impacts["reputation"] = 0.5 * efficiency
    elif dept_name == "Operations":
        impacts["burn_rate"] = -10_000 * efficiency
        impacts["technical_debt"] = 1.0 * efficiency
    elif dept_name == "RD":
        impacts["innovation_index"] = 4.0 * efficiency
        impacts["technical_debt"] = 2.0 * efficiency
    elif dept_name == "Legal":
        impacts["reputation"] = 1.0 * efficiency

    new_state_dict = state.model_dump()
    for k, v in impacts.items():
        if k in new_state_dict:
            new_state_dict[k] += v

    for k in [
        "market_share",
        "reputation",
        "product_quality",
        "technical_debt",
        "innovation_index",
        "market_sentiment",
        "employee_happiness",
    ]:
        new_state_dict[k] = max(0.0, min(100.0, new_state_dict[k]))

    new_state = CompanyState(**new_state_dict)

    result_str = "SUCCESS" if is_success else "UNDERPERFORMED"
    log_msg = (
        f"{dept_name}: {output.plan} -> {result_str} (Cost: ${actual_cost:,.0f}){msg}"
    )

    return new_state.log(log_msg)


# ==============================================================================
# 5. Node Definitions (System 1 vs System 2)
# ==============================================================================


def create_ceo_node() -> ProbabilisticNode:
    return ProbabilisticNode(
        name="CEO",
        adk_config=adk_config,
        output_schema=CEOOutput,
        state_type=CompanyState,
        prompt_template="""CORPORATE COMMAND (SYSTEM 1 PLANNER)
        Quarter: {{quarter}} | Budget: ${{budget:,.0f}} | Burn: ${{burn_rate:,.0f}}

        MARKET EVENT: {{last_event}}

        KPIs:
        - Market Share: {{market_share|round(1)}}%
        - Quality: {{product_quality|round(1)}}
        - Innovation: {{innovation_index|round(1)}}
        - Tech Debt: {{technical_debt|round(1)}}
        - Sentiment: {{market_sentiment|round(1)}}

        {% if board_warning %}
        🚨 BOARD REJECTION FEEDBACK: {{board_warning}}
        🚨 YOU MUST ADJUST YOUR PROPOSAL TO ADDRESS THIS!
        {% endif %}

        Analyze the situation. We need to survive and grow.
        Select ONE department to address the most critical issue.
        Allocate budget wisely (Available: ${{budget:,.0f}}).
        """,
        state_updater=lambda s, r: s.log(
            f"CEO: Proposes {r.department} with ${r.budget_allocation:,.0f} - {r.analysis[:40]}..."
        ).update(
            proposed_department=r.department,
            proposed_budget=r.budget_allocation,
            ceo_instruction=r.instruction,
            board_warning="",  # Clear old warning
            next_action="Board",
        ),
    )


def create_board_node() -> ProbabilisticNode:
    def board_updater(s: CompanyState, r: BoardOutput) -> CompanyState:
        if r.approved:
            return s.log(f"BOARD: ✅ Approved. {r.feedback_for_ceo[:60]}...").update(
                next_action=s.proposed_department
            )
        return s.log(f"BOARD: ❌ REJECTED! {r.feedback_for_ceo[:60]}...").update(
            board_warning=r.feedback_for_ceo, next_action="CEO"
        )

    return ProbabilisticNode(
        name="Board",
        adk_config=adk_config,
        output_schema=BoardOutput,
        state_type=CompanyState,
        prompt_template="""BOARD OF DIRECTORS (SYSTEM 2 CRITIC)

        Quarter: {{quarter}} | Budget: ${{budget:,.0f}} | Burn: ${{burn_rate:,.0f}}

        CEO PROPOSAL:
        - Target Department: {{proposed_department}}
        - Requested Budget: ${{proposed_budget:,.0f}}
        - Instruction: {{ceo_instruction}}

        KPIs:
        - Market Share: {{market_share|round(1)}}%
        - Quality: {{product_quality|round(1)}}
        - Tech Debt: {{technical_debt|round(1)}}

        Your job is to prevent the CEO from making poor decisions.
        If the budget request is dangerously high relative to the total budget, REJECT it.
        If tech debt is > 40 and the CEO is not funding Engineering, REJECT it.
        If the plan is sound, APPROVE it. Provide actionable feedback.
        """,
        state_updater=board_updater,
    )


def create_dept_node(name: str, mission: str) -> ProbabilisticNode:
    return ProbabilisticNode(
        name=name,
        adk_config=adk_config,
        output_schema=DeptOutput,
        state_type=CompanyState,
        prompt_template=f"""DEPARTMENT: {name}
        MISSION: {mission}

        CEO INSTRUCTION: {{{{ceo_instruction}}}}
        ALLOCATED BUDGET: ${{{{proposed_budget:,.0f}}}}

        CURRENT STATE:
        Total Budget: ${{{{budget:,.0f}}}}
        Market Share: {{{{market_share}}}}%
        Quality: {{{{product_quality}}}}

        Propose a concrete action plan. Estimate the cost carefully.
        Do not exceed your allocated budget of ${{{{proposed_budget:,.0f}}}}.
        """,
        state_updater=lambda s, r: simulate_department_action(s, name, r),
    )


# ==============================================================================
# 6. Simulation Engine
# ==============================================================================


def build_graph() -> Graph:
    graph = Graph(name="CorpCommand", state_type=CompanyState)

    ceo = create_ceo_node()
    board = create_board_node()
    graph.add_node(ceo)
    graph.add_node(board)
    graph.entry_point = "CEO"

    # System 1 -> System 2
    graph.add_transition("CEO", "Board", condition=lambda s: s.next_action == "Board")

    # System 2 Loopback
    graph.add_transition("Board", "CEO", condition=lambda s: s.next_action == "CEO")

    depts = {
        "Marketing": "Drive growth and visibility.",
        "Engineering": "Build product and fix debt.",
        "HR": "Recruit and retain talent.",
        "Operations": "Optimize efficiency and costs.",
        "RD": "Invent the future.",
        "Legal": "Protect the company.",
    }

    for name, mission in depts.items():
        node = create_dept_node(name, mission)
        graph.add_node(node)
        # System 2 -> Executor (if approved)
        graph.add_transition(
            "Board", name, condition=lambda s, n=name: s.next_action == n
        )

    return graph


# ==============================================================================
# 7. Visualization
# ==============================================================================


def create_dashboard(state: CompanyState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="logs", size=12),
    )

    layout["header"].update(
        Panel(
            f"[bold blue]CORPORATE DIGITAL TWIN (SYS1+SYS2)[/bold blue] - Q{state.quarter} Turn {state.turn}",
            style="white on blue",
        )
    )

    stats = Table(title="Live Telemetry", expand=True)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="magenta")

    stats.add_row("💰 Budget", f"${state.budget:,.0f}")
    stats.add_row("🔥 Burn Rate", f"${state.burn_rate:,.0f}")
    stats.add_row("📈 Revenue", f"${state.revenue:,.0f}")
    stats.add_row("🌍 Market Share", f"{state.market_share:.1f}%")
    stats.add_row("⭐ Product Quality", f"{state.product_quality:.1f}")
    stats.add_row("📉 Tech Debt", f"{state.technical_debt:.1f}")
    stats.add_row("😊 Employee Joy", f"{state.employee_happiness:.1f}")

    layout["main"].update(Panel(stats))

    log_text = Text()
    for log in state.logs[-10:]:
        if "CEO:" in log:
            log_text.append(f"{log}\n", style="bold yellow")
        elif "BOARD: ✅" in log:
            log_text.append(f"{log}\n", style="bold green")
        elif "BOARD: ❌" in log:
            log_text.append(f"{log}\n", style="bold red blink")
        elif "SUCCESS" in log:
            log_text.append(f"{log}\n", style="green")
        elif "UNDERPERFORMED" in log:
            log_text.append(f"{log}\n", style="red")
        else:
            log_text.append(f"{log}\n", style="dim")

    layout["logs"].update(Panel(log_text, title="Action Log"))

    return layout


# ==============================================================================
# 8. Main
# ==============================================================================


async def main():
    console.clear()
    console.print("[bold green]Initializing Corporate Simulation...[/bold green]")

    graph = build_graph()
    state = CompanyState()

    total_quarters = 2
    turns_per_quarter = 2

    with Live(create_dashboard(state), refresh_per_second=4, screen=True) as live:
        for q in range(1, total_quarters + 1):
            state = state.update(quarter=q)

            revenue = (state.market_share * 100_000) * (state.product_quality / 50.0)
            state = state.update(
                budget=state.budget + revenue - (state.burn_rate * 3),
                revenue=state.revenue + revenue,
            )
            state = state.log(
                f"--- Q{q} Financials: +${revenue:,.0f} Rev, -${state.burn_rate * 3:,.0f} Burn ---"
            )

            for t in range(1, turns_per_quarter + 1):
                state = state.update(turn=t)
                live.update(create_dashboard(state))
                await asyncio.sleep(0.5)

                event, impacts = generate_market_event()
                state = state.log(f"⚡ EVENT: {event}")
                s_dict = state.model_dump()
                for k, v in impacts.items():
                    if k in s_dict:
                        s_dict[k] += v
                state = CompanyState(**s_dict).update(last_event=event)
                live.update(create_dashboard(state))
                await asyncio.sleep(0.5)

                if state.budget < -500_000:
                    state = state.log("💀 BANKRUPTCY DECLARED.")
                    live.update(create_dashboard(state))
                    return

                try:
                    # Run the governance loop
                    state = await graph.run(state)
                except Exception as e:
                    state = state.log(f"❌ Error: {e}")

                live.update(create_dashboard(state))
                await asyncio.sleep(1.0)

            state = state.log(f"End of Q{q}. Market evolving...")
            state = state.update(
                technical_debt=state.technical_debt + 2.0,
                product_quality=state.product_quality - 1.0,
            )

    console.print("[bold green]Simulation Complete![/bold green]")
    console.print(f"Final Budget: ${state.budget:,.2f}")
    console.print(f"Market Share: {state.market_share:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
