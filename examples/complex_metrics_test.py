import asyncio
import random
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode
from markov_agent.topology.graph import Graph
from markov_agent.topology.edge import Edge
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.simulation.metrics import calculate_metrics

"""
Complex Metrics Test (Pass@k & Pass^k)

This script demonstrates the difference between:
- Accuracy (Mean Pass Rate / pass@1)
- Consistency (pass^k): Reliability of getting the SAME result every time.
- Reliability (pass@k): Probability of getting AT LEAST ONE correct result given k attempts.

It uses a simulated 'MathSolverNode' with varying difficulty levels to generate
stochastic results, then runs a Monte Carlo simulation to compute metrics.
"""

console = Console()

# 1. Define State
class MathState(BaseState):
    problem_id: str
    difficulty: float  # 0.0 to 1.0 (probability of failure)
    answer: int | None = None
    is_correct: bool = False

# 2. Define a Stochastic Node (Simulating an LLM)
class MathSolverNode(BaseNode[MathState]):
    async def execute(self, state: MathState) -> MathState:
        # Simulate probabilistic failure based on difficulty
        # Random float 0.0 to 1.0. If > difficulty, success.
        # This is a simplification.
        
        # We use random.random() here. Since we run parallel in simulation, 
        # distinct runs will get different random values.
        roll = random.random()
        success = roll > state.difficulty
        
        state.is_correct = success
        state.answer = 42 if success else 0
        state.record_step({"node": self.name, "roll": roll, "success": success})
        return state

# 3. Setup Simulation
async def run_demo():
    console.print("[bold blue]Running Complex Metrics Test (Pass@k & Pass^k)[/bold blue]")

    # Topology
    solver = MathSolverNode(name="solver")
    # Simple graph: Start -> Solver -> End
    # Edge logic: Always go to END
    edge = Edge(source=solver.name, target_func=lambda s: "END")
    
    graph = Graph(nodes={solver.name: solver}, edges=[edge], entry_point=solver.name)

    # Dataset: A mix of difficulties
    # 2 Deterministic cases (0% fail rate) - Should be Consistent
    # 5 Easy cases (10% fail rate)
    # 5 Hard cases (90% fail rate)
    # 5 Medium cases (50% fail rate)
    dataset = []
    
    for i in range(2):
        dataset.append(MathState(problem_id=f"det_{i}", difficulty=0.0))
    for i in range(5):
        dataset.append(MathState(problem_id=f"easy_{i}", difficulty=0.1))
    for i in range(5):
        dataset.append(MathState(problem_id=f"medium_{i}", difficulty=0.5))
    for i in range(5):
        dataset.append(MathState(problem_id=f"hard_{i}", difficulty=0.9))

    console.print(f"Dataset size: {len(dataset)} cases.")
    
    # Run Simulation
    # We want enough samples to see pass@k diffs. Let's do n=20 samples per case.
    n_runs = 20
    console.print(f"Running Monte Carlo Simulation with n_runs={n_runs}...")
    
    runner = MonteCarloRunner(
        graph=graph,
        dataset=dataset,
        n_runs=n_runs,
        success_criteria=lambda s: s.is_correct
    )
    
    results = await runner.run_simulation()
    
    # Calculate Metrics
    metrics = calculate_metrics(results)
    
    # Display Results
    print_metrics_table(metrics)

def print_metrics_table(metrics: dict[str, Any]):
    table = Table(title="Simulation Metrics Report")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")

    table.add_row(
        "Accuracy (pass@1)", 
        f"{metrics['accuracy']:.2%}", 
        "Mean success rate across all runs"
    )
    table.add_row(
        "Consistency (pass^k)", 
        f"{metrics['consistency']:.2%}", 
        "Strict Consistency: % of cases that succeeded 100% of the time"
    )
    table.add_row(
        "Reliability (pass@n)", 
        f"{metrics['reliability']:.2%}", 
        "At least one success in n runs"
    )

    table.add_section()
    
    # Pass@k Estimates
    pass_at_k = metrics.get('pass_at_k', {})
    for k_key, score in pass_at_k.items():
        table.add_row(
            f"{k_key} Estimate", 
            f"{score:.2%}", 
            f"Unbiased estimator for {k_key}"
        )

    console.print(table)
    console.print("\n[dim]Note: 'pass^k' in this context is labeled as Consistency (probability that ALL samples are correct).[/dim]")

if __name__ == "__main__":
    asyncio.run(run_demo())
