import asyncio

import rich

from markov_agent.core.state import BaseState
from markov_agent.simulation.analysis import TransitionMatrixAnalyzer
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class DemoState(BaseState):
    """Simple state for transition matrix demonstration."""

    value: int = 0


class NoOpNode(BaseNode[DemoState]):
    """A node that does nothing but pass the state."""

    async def execute(self, state: DemoState) -> DemoState:
        return state


async def run_demo():
    console = rich.get_console()
    console.print("[bold cyan]Transition Matrix Analysis Demo[/bold cyan]")

    # Define nodes
    node_start = NoOpNode(name="START")
    node_branch = NoOpNode(name="BRANCH")
    node_end = NoOpNode(name="END")

    # Define probabilistic transitions
    # START -> BRANCH (100%)
    # BRANCH -> START (20%) - Loop back
    # BRANCH -> END (80%)   - Terminal

    def branch_router(state: DemoState) -> dict[str, float]:
        return {"START": 0.2, "END": 0.8}

    graph = Graph(
        name="demo_graph",
        nodes={"START": node_start, "BRANCH": node_branch, "END": node_end},
        edges=[
            Edge(source="START", target_func=lambda _: "BRANCH"),
            Edge(source="BRANCH", target_func=branch_router),
        ],
        entry_point="START",
        state_type=DemoState,
    )

    # Run Monte Carlo Simulation
    console.print("\nRunning 50 Monte Carlo simulations...")
    runner = MonteCarloRunner(graph=graph, dataset=[DemoState()], n_runs=50)

    results = await runner.run_simulation()

    # Build Transition Matrix
    analyzer = TransitionMatrixAnalyzer(results)
    matrix = analyzer.to_dataframe()

    console.print("\n[bold green]Empirical Transition Matrix:[/bold green]")
    console.print(matrix)

    console.print(
        "\n[dim]Note: Matrix values represent the probability P(next|current) observed during simulation.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
