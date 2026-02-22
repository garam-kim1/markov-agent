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
    level="WARNING",  # Reduce noise from libraries
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("company_sim")
console = Console()

# ==============================================================================
# 1. Configuration
# ==============================================================================

# Use environment variable or default to Gemini, but keep local fallback if needed
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"  # Fast, good for simulation

adk_config = ADKConfig(
    model_name=MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7,
    max_input_tokens=9000,
    # If no API key, try local fallback (simulated for example portability)
    use_litellm=True if not API_KEY else False,
)

if not API_KEY:
    console.print(
        "[yellow]No GEMINI_API_KEY found. Attempting to use local/default config (might fail if not set up).[/yellow]"
    )
    # Fallback to the original local config for the user who had it
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
    market_share: float = Field(default=5.0)  # Percentage
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


class DeptOutput(BaseModel):
    plan: str = Field(description="The action plan.")
    expected_cost: float
    focus_area: str = Field(description="Primary metric to improve.")


# ==============================================================================
# 4. Logic & Simulation (The Physics)
# ==============================================================================


def generate_market_event() -> tuple[str, dict[str, float]]:
    """Generate a random market event."""
    events = [
        ("Tech Boom", {"market_sentiment": 10.0, "competitor_aggression": 5.0}),
        ("Market Crash", {"market_sentiment": -15.0, "budget": -50000.0}),
        ("Viral Competitor Product", {"market_share": -2.0, "reputation": -5.0}),
        ("Regulatory Crackdown", {"technical_debt": 5.0, "budget": -20000.0}),
        ("Talent Shortage", {"employee_happiness": -10.0, "burn_rate": 20000.0}),
        ("Stable Market", {"market_sentiment": 1.0}),
        ("Stable Market", {"market_sentiment": -1.0}),
    ]
    # Weighted random choice
    weights = [0.1, 0.05, 0.1, 0.05, 0.1, 0.3, 0.3]
    event, impacts = random.choices(events, weights=weights, k=1)[0]
    return event, impacts


def simulate_department_action(
    state: CompanyState, dept_name: str, output: DeptOutput
) -> CompanyState:
    """Deterministically (with noise) calculate the result of a department's action."""

    # 1. Base Success Rate
    success_chance = 0.7 + (state.employee_happiness / 500.0)  # 0.7 + 0.15 = 0.85 max
    if state.budget < 0:
        success_chance -= 0.3

    roll = random.random()
    is_success = roll < success_chance

    # 2. Calculate Impacts
    impacts = {}
    msg = ""

    # Cap cost at budget
    actual_cost = min(output.expected_cost, state.budget)
    if actual_cost < output.expected_cost:
        msg = f" (Budget constrained: req ${output.expected_cost / 1000:.0f}k -> got ${actual_cost / 1000:.0f}k)"

    impacts["budget"] = -actual_cost

    efficiency = (actual_cost / 100_000.0) * (1.5 if is_success else 0.5)

    if dept_name == "Marketing":
        impacts["market_share"] = 1.5 * efficiency
        impacts["market_sentiment"] = 2.0 * efficiency
        impacts["reputation"] = 1.0 * efficiency
    elif dept_name == "Engineering":
        impacts["product_quality"] = 2.0 * efficiency
        impacts["technical_debt"] = -3.0 * efficiency
        impacts["burn_rate"] = -5000 * efficiency  # Optimization
    elif dept_name == "HR":
        impacts["employee_happiness"] = 5.0 * efficiency
        impacts["reputation"] = 0.5 * efficiency
    elif dept_name == "Operations":
        impacts["burn_rate"] = -10_000 * efficiency
        impacts["technical_debt"] = 1.0 * efficiency  # Cutting corners?
    elif dept_name == "RD":
        impacts["innovation_index"] = 4.0 * efficiency
        impacts["technical_debt"] = 2.0 * efficiency  # New exp code
    elif dept_name == "Legal":
        # Defensive
        impacts["reputation"] = 1.0 * efficiency

    # 3. Apply Impacts
    new_state_dict = state.model_dump()
    for k, v in impacts.items():
        if k in new_state_dict:
            new_state_dict[k] += v

    # Clamp values
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
# 5. Node Definitions
# ==============================================================================


def create_ceo_node() -> ProbabilisticNode:
    return ProbabilisticNode(
        name="CEO",
        adk_config=adk_config,
        output_schema=CEOOutput,
        state_type=CompanyState,
        prompt_template="""CORPORATE DASHBOARD
        Quarter: {{quarter}} | Budget: ${{budget:,.0f}} | Burn: ${{burn_rate:,.0f}}

        MARKET EVENT: {{last_event}}

        KPIs:
        - Market Share: {{market_share|round(1)}}%
        - Quality: {{product_quality|round(1)}}
        - Innovation: {{innovation_index|round(1)}}
        - Tech Debt: {{technical_debt|round(1)}}
        - Sentiment: {{market_sentiment|round(1)}}

        Analyze the situation. We need to survive and grow.
        Select ONE department to address the most critical issue.
        Allocate budget wisely (Available: ${{budget:,.0f}}).
        """,
        # CEO doesn't update state directly, just decides routing
        state_updater=lambda s, r: s.log(
            f"CEO: {r.analysis} -> Activating {r.department} with ${r.budget_allocation:,.0f}"
        ).update(next_action=r.department),
    )


def create_dept_node(name: str, mission: str) -> ProbabilisticNode:
    return ProbabilisticNode(
        name=name,
        adk_config=adk_config,
        output_schema=DeptOutput,
        state_type=CompanyState,
        prompt_template=f"""DEPARTMENT: {name}
        MISSION: {mission}

        CEO INSTRUCTION: {{logs[-1]}}

        CURRENT STATE:
        Budget: ${{{{budget:,.0f}}}}
        Market Share: {{{{market_share}}}}%
        Quality: {{{{product_quality}}}}

        Propose a concrete action plan. Estimate the cost carefully.
        """,
        state_updater=lambda s, r: simulate_department_action(s, name, r),
    )


# ==============================================================================
# 6. Simulation Engine
# ==============================================================================


def build_graph() -> Graph:
    """Build the corporate command graph."""
    graph = Graph(name="CorpCommand", state_type=CompanyState)

    ceo = create_ceo_node()
    graph.add_node(ceo)
    graph.entry_point = "CEO"

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

        # CEO -> Dept
        graph.add_transition(
            "CEO", name, condition=lambda s, n=name: s.next_action == n
        )

        # Dept -> Terminal (End of Turn)
        # We do NOT add a transition back to CEO here.
        # This makes one run() call equal to one command cycle.

    return graph


# ==============================================================================
# 7. Visualization
# ==============================================================================


def create_dashboard(state: CompanyState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="logs", size=10),
    )

    layout["header"].update(
        Panel(
            f"[bold blue]CORPORATE DIGITAL TWIN[/bold blue] - Q{state.quarter} Turn {state.turn}",
            style="white on blue",
        )
    )

    # Stats Table
    stats = Table(title="Company Metrics", expand=True)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="magenta")

    stats.add_row("💰 Budget", f"${state.budget:,.0f}")
    stats.add_row("🔥 Burn Rate", f"${state.burn_rate:,.0f}")
    stats.add_row("📈 Revenue", f"${state.revenue:,.0f}")
    stats.add_row("🌍 Market Share", f"{state.market_share:.1f}%")
    stats.add_row("⭐ Product Quality", f"{state.product_quality:.1f}")
    stats.add_row("💡 Innovation", f"{state.innovation_index:.1f}")
    stats.add_row("📉 Tech Debt", f"{state.technical_debt:.1f}")
    stats.add_row("😊 Employee Joy", f"{state.employee_happiness:.1f}")

    layout["main"].update(Panel(stats, title="Live Telemetry"))

    # Logs
    log_text = Text()
    for log in state.logs[-8:]:
        if "CEO:" in log:
            log_text.append(f"{log}\n", style="bold yellow")
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

    # Run for 4 Quarters, 3 Turns per Quarter
    total_quarters = 4
    turns_per_quarter = 3

    with Live(create_dashboard(state), refresh_per_second=4, screen=True) as live:
        for q in range(1, total_quarters + 1):
            state = state.update(quarter=q)

            # Quarterly Burn/Revenue Check
            revenue = (state.market_share * 100_000) * (state.product_quality / 50.0)
            state = state.update(
                budget=state.budget
                + revenue
                - (state.burn_rate * 3),  # 3 months roughly
                revenue=state.revenue + revenue,
            )
            state = state.log(
                f"--- Q{q} Financials: +${revenue:,.0f} Rev, -${state.burn_rate * 3:,.0f} Burn ---"
            )

            for t in range(1, turns_per_quarter + 1):
                state = state.update(turn=t)
                live.update(create_dashboard(state))
                await asyncio.sleep(1.0)  # Pause for readability

                # 1. Market Event
                event, impacts = generate_market_event()
                state = state.log(f"⚡ EVENT: {event}")
                # Apply event impacts
                s_dict = state.model_dump()
                for k, v in impacts.items():
                    if k in s_dict:
                        s_dict[k] += v
                state = CompanyState(**s_dict).update(last_event=event)
                live.update(create_dashboard(state))
                await asyncio.sleep(1.0)

                # 2. Check Game Over
                if state.budget < -500_000:
                    state = state.log("💀 BANKRUPTCY DECLARED.")
                    live.update(create_dashboard(state))
                    return

                # 3. AI Turn (CEO -> Dept)
                try:
                    # Run the graph (CEO -> Dept -> Stop)
                    # Graph.run() executes from entry_point until terminal node
                    state = await graph.run(state)
                except Exception as e:
                    state = state.log(f"❌ Error: {e}")

                live.update(create_dashboard(state))
                await asyncio.sleep(1.5)  # Let user read

            # End of Quarter Evolution
            state = state.log(f"End of Q{q}. Market evolving...")
            # Entropy
            state = state.update(
                technical_debt=state.technical_debt + 2.0,
                product_quality=state.product_quality - 1.0,
            )

    console.print("[bold green]Simulation Complete![/bold green]")
    console.print(f"Final Budget: ${state.budget:,.2f}")
    console.print(f"Market Share: {state.market_share:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
