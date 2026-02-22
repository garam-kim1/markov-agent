import asyncio
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from markov_agent import (
    ADKConfig,
    BaseDigitalTwin,
    BaseState,
    Graph,
    MonteCarloRunner,
    TopologyOptimizer,
)

console = Console()


# 1. Define the State
class AdaptiveSystemState(BaseState):
    balance: float = 100.0
    capacity: int = 10
    requested_withdrawal: float = 0.0
    status: str = "IDLE"


class MarketDecision(BaseModel):
    decision: str
    amount: float


# 2. Define the Digital Twin (The "Shell" / Physical Laws)
class SystemDigitalTwin(BaseDigitalTwin):
    async def validate_transition(
        self, current: AdaptiveSystemState, proposed: AdaptiveSystemState
    ) -> bool:
        # Business Law: Cannot have negative balance
        if proposed.balance < 0:
            console.print("[red]Twin: Violation! Negative balance detected.[/red]")
            return False
        # Physical Law: Capacity cannot be exceeded
        if proposed.capacity > 20:
            console.print("[red]Twin: Violation! Capacity limit reached.[/red]")
            return False
        return True

    async def calculate_consequences(
        self, state: AdaptiveSystemState, action: Any
    ) -> AdaptiveSystemState:
        # World Model: Withdrawal action reduces balance
        if action.get("type") == "withdraw":
            amount = action.get("amount", 0)
            return state.update(
                balance=state.balance - amount, requested_withdrawal=amount
            )
        return state


# 3. Build the Agent Graph
async def run_demo():
    # Setup Mock LLM for deterministic behavior in demo
    mock_responses = {
        "Should I withdraw?": '{"decision": "YES", "amount": 150}',  # Will fail validation
        "Try again?": '{"decision": "YES", "amount": 50}',  # Will succeed
    }

    adk_config = ADKConfig(model_name="mock-model")

    # Setup mock responder
    def responder(p):
        return mock_responses.get(
            next((k for k in mock_responses if k in p), "DEFAULT"),
            '{"decision": "NO", "amount": 0.0}',
        )

    g = Graph(
        "AdaptiveAgent", state_type=AdaptiveSystemState, default_adk_config=adk_config
    )

    @g.node(mock_responder=responder)
    async def analyze_market(state: AdaptiveSystemState) -> MarketDecision:
        """Should I withdraw? Output JSON with 'decision' (YES/NO) and 'amount'."""
        return MarketDecision(decision="NO", amount=0.0)

    @g.task()
    def process_decision(state: AdaptiveSystemState) -> AdaptiveSystemState:
        # Access the last node output from history
        last_output = state.history[-1].get("output", {})
        decision = last_output.get("decision")
        amount = last_output.get("amount", 0.0)

        if decision == "YES":
            return state.update(requested_withdrawal=amount, status="PENDING")
        return state.update(status="IDLE")

    @g.task()
    async def validate_with_twin(state: AdaptiveSystemState) -> AdaptiveSystemState:
        twin = SystemDigitalTwin()
        # Create a proposed state
        proposed = state.update(balance=state.balance - state.requested_withdrawal)
        if await twin.validate_transition(state, proposed):
            return proposed.update(status="SUCCESS")
        return state.update(status="REJECTED")

    # Transitions
    g.add_transition("analyze_market", "process_decision")
    g.route(
        "process_decision",
        {
            "validate_with_twin": lambda s: s.status == "PENDING",
            "analyze_market": lambda s: s.status == "IDLE",
        },
    )

    # 4. Simulation & Optimization
    console.print("\n[bold cyan]Step 1: Running Initial Simulation[/bold cyan]")
    dataset = [AdaptiveSystemState(balance=100)]
    runner = MonteCarloRunner(
        graph=g,
        dataset=dataset,
        n_runs=5,
        success_criteria=lambda s: s.status == "SUCCESS",
    )
    results = await runner.run_simulation()

    # 5. Topology Optimization
    console.print("\n[bold cyan]Step 2: Optimizing Topology[/bold cyan]")
    optimizer = TopologyOptimizer(g)

    # Detect high entropy/failure nodes
    candidates = optimizer.suggest_fission(results)
    console.print(f"Candidates for Node Fission: {candidates}")

    # Prune failing paths (demonstration)
    removed = optimizer.prune_edges(results, threshold=0.1)
    console.print(f"Pruned {removed} underperforming edges.")

    # 6. Display Results
    table = Table(title="Simulation Results")
    table.add_column("Case")
    table.add_column("Final Balance")
    table.add_column("Status")
    table.add_column("Success")

    for res in results:
        final_bal = res.final_state.balance if res.final_state else "ERROR"
        final_status = res.final_state.status if res.final_state else "ERROR"
        table.add_row(
            res.case_id,
            f"{final_bal}",
            final_status,
            "[green]YES[/green]" if res.success else "[red]NO[/red]",
        )

    console.print(table)

    console.print("\n[bold green]Neuro-Symbolic Foundation established.[/bold green]")
    console.print(
        "The Agent now operates within the constraints of its Digital Twin (Shell),"
    )
    console.print("and its Topology evolves based on simulation feedback.")


if __name__ == "__main__":
    asyncio.run(run_demo())
