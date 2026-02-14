import asyncio

import rich
from rich.table import Table

from markov_agent.core.state import BaseState
from markov_agent.topology.analysis import TopologyAnalyzer
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class ResearchState(BaseState):
    """State for a simulated research agent."""

    topic: str = "Markov Chains"
    depth: int = 0
    uncertainty: float = 0.5


class ActionNode(BaseNode[ResearchState]):
    """Simulates an action that might increase depth or change uncertainty."""

    async def execute(self, state: ResearchState) -> ResearchState:
        # Simulate some work
        new_state = state.update(depth=state.depth + 1)
        new_state.record_step({"node": self.name})
        return new_state


async def run_topology_demo():
    console = rich.get_console()
    console.print(
        "[bold cyan]Advanced Topology & Probability Analysis Demo[/bold cyan]"
    )

    # 1. Define a Graph with Probabilistic Branching
    node_start = ActionNode(name="START", state_type=ResearchState)
    node_research = ActionNode(name="RESEARCH", state_type=ResearchState)
    node_verify = ActionNode(name="VERIFY", state_type=ResearchState)
    node_publish = ActionNode(name="PUBLISH", state_type=ResearchState)

    def research_router(state: ResearchState):
        # 3-way branch with varied entropy
        return {
            "RESEARCH": 0.4,  # Continue researching
            "VERIFY": 0.5,  # Move to verification
            "PUBLISH": 0.1,  # Early publish (unlikely)
        }

    def verify_router(state: ResearchState):
        # High uncertainty branch
        return {"RESEARCH": 0.5, "PUBLISH": 0.5}

    graph = Graph(
        name="research_agent",
        nodes={
            "START": node_start,
            "RESEARCH": node_research,
            "VERIFY": node_verify,
            "PUBLISH": node_publish,
        },
        edges=[
            Edge(source="START", target_func=lambda _: "RESEARCH"),
            Edge(source="RESEARCH", target_func=research_router),
            Edge(source="VERIFY", target_func=verify_router),
        ],
        entry_point="START",
        state_type=ResearchState,
        strict_markov=True,  # Enforce Markov property
    )

    # 2. Analyze the Topology Statically
    console.print("\n[bold green]1. Static Topology Analysis[/bold green]")
    analyzer = TopologyAnalyzer(graph)
    matrix = analyzer.extract_matrix(sample_count=100)

    # Display Matrix
    table = Table(title="Transition Matrix (Empirical Approximation)")
    table.add_column(r"Source \ Target")
    for node in analyzer.nodes:
        table.add_column(node)

    for i, source in enumerate(analyzer.nodes):
        row = [source]
        for j in range(len(analyzer.nodes)):
            row.append(f"{matrix[i, j]:.2f}")
        table.add_row(*row)
    console.print(table)

    absorbing = analyzer.detect_absorbing_states(matrix)
    console.print(f"Absorbing States: [bold yellow]{absorbing}[/bold yellow]")

    stationary = analyzer.calculate_stationary_distribution(matrix)
    console.print("\nStationary Distribution (Long-term probability):")
    for node, prob in zip(analyzer.nodes, stationary, strict=True):
        console.print(f"  {node}: {prob:.4f}")

    # NEW: Advanced Markov Diagnostics
    console.print("\n[bold green]2. Advanced Markov Diagnostics[/bold green]")
    is_ergodic = analyzer.is_ergodic(matrix)
    console.print(f"Is Ergodic (Irreducible & Aperiodic): [bold]{is_ergodic}[/bold]")

    mixing_time = analyzer.calculate_mixing_time(matrix)
    console.print(
        f"Mixing Time (approx. steps to equilibrium): [bold]{mixing_time}[/bold]"
    )

    # Calculate probability of a specific trajectory
    sample_traj = ["START", "RESEARCH", "VERIFY", "PUBLISH"]
    traj_prob = analyzer.simulate_trajectory_probability(sample_traj, matrix)
    console.print(
        f"Likelihood of trajectory {sample_traj}: [bold]{traj_prob:.4f}[/bold]"
    )

    # Generate Mermaid Diagram
    console.print("\n[bold green]3. Topology Visualization (Mermaid)[/bold green]")
    mermaid = analyzer.generate_mermaid_graph(matrix, threshold=0.05)
    console.print("Copy-paste into Mermaid Live Editor:")
    console.print(f"[dim]\n{mermaid}\n[/dim]")

    # 3. Demonstrate Log-Space Probability and Beam Search
    console.print("\n[bold green]4. Beam Search & Log-Space Probability[/bold green]")
    initial_state = ResearchState(topic="Topology Engineering")

    # Run beam search to find top 3 most likely trajectories
    trajectories = await graph.run_beam(initial_state, width=3, max_steps=5)

    for i, traj in enumerate(trajectories):
        conf = traj.meta.get("confidence", 0.0)
        log_p = traj.meta.get("cumulative_log_prob", -float("inf"))

        console.print(
            f"\n[bold]Trajectory #{i + 1}[/bold] (p={conf:.4f}, log_p={log_p:.2f})"
        )
        path = " -> ".join([h["node"] for h in traj.history])
        console.print(f"  Path: {path}")

        # Show entropy history if available
        if "step_entropy" in traj.meta:
            entropy_str = " -> ".join([f"{e:.2f}" for e in traj.meta["step_entropy"]])
            console.print(f"  Entropy Profile: {entropy_str}")

    # 4. Demonstrate High Uncertainty Logging
    console.print(
        "\n[bold green]3. Live Execution with Uncertainty Tracking[/bold green]"
    )
    # Run a single path
    final_state = await graph.run(initial_state)
    console.print(f"\nFinal State Depth: {final_state.depth}")
    console.print(f"Final Confidence: {final_state.meta.get('confidence'):.4f}")


if __name__ == "__main__":
    asyncio.run(run_topology_demo())
