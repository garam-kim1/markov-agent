import asyncio
import random
from typing import Any

from rich.console import Console
from rich.table import Table

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode

"""
Complex Metrics Test (Pass@k & Pass^k) - Code Generation Edition

This script simulates a "Code Generation Agent" evaluating its performance on a dataset.
- Accuracy (pass@1): How often it gets it right on the first try.
- Consistency (pass^k): Probability that k random attempts are ALL correct.
- Reliability (pass@k): Probability that at least ONE of k attempts is correct.

We use a ProbabilisticNode with a mock responder that fails stochastically based
on the problem's complexity.
"""

console = Console()


# 1. Define State for Code Generation
class CodeState(BaseState):
    problem_id: str
    complexity: float  # 0.0 (trivial) to 1.0 (impossible)
    generated_code: str | None = None
    is_correct: bool = False


# 2. Mock Responder Logic
# Simulates an LLM that is less reliable as complexity increases.
def code_gen_mock_responder(prompt: str) -> dict:
    # This is a bit of a hack because ProbabilisticNode's mock_responder
    # doesn't easily see the state's complexity directly from the adk_controller.
    # We'll encode complexity in the prompt for the mock to "see" it.
    if "COMPLEXITY:" in prompt:
        try:
            complexity = float(prompt.split("COMPLEXITY:")[1].split()[0])
        except (ValueError, IndexError):
            complexity = 0.5
    else:
        complexity = 0.5

    # Probability of success = 1.0 - complexity
    success_rate = 1.0 - complexity
    is_success = random.random() < success_rate

    if is_success:
        return {"code": "def solution(): return True", "valid": True}
    return {"code": "def solution(): raise Exception()", "valid": False}


# 3. State Updater for Code Generation
def update_code_state(state: CodeState, result: Any) -> CodeState:
    # 'result' comes from the mock responder
    new_state = state.model_copy(deep=True)
    new_state.generated_code = result.get("code")
    new_state.is_correct = result.get("valid", False)
    new_state.record_step({"node": "coder", "success": new_state.is_correct})
    return new_state


async def run_demo():
    console.print(
        "[bold blue]Running Code Generation Simulation: Pass@k vs Pass^k[/bold blue]",
    )

    # Topology setup
    coder = ProbabilisticNode(
        name="code_generator",
        adk_config=ADKConfig(model_name="mock-coder", temperature=0.8),
        prompt_template="Generate code for: {{ problem_id }}. COMPLEXITY: {{ complexity }}",
        mock_responder=code_gen_mock_responder,
        state_updater=update_code_state,
        samples=1,  # 1 sample per node execution, multiple runs per case in simulation
    )

    # Explicit terminal node
    class EndNode(BaseNode[CodeState]):
        async def execute(self, state: CodeState) -> CodeState:
            return state

    end = EndNode(name="END")

    edge = Edge(source=coder.name, target_func=lambda s: "END")
    graph = Graph(
        name="metrics_graph",
        nodes={coder.name: coder, "END": end},
        edges=[edge],
        entry_point=coder.name,
    )

    # Dataset: Various complexities
    dataset = []
    # 2 "Hello World" level (0.05 complexity)
    for i in range(2):
        dataset.append(CodeState(problem_id=f"trivial_{i}", complexity=0.05))
    # 5 "Easy Leetcode" (0.3 complexity)
    for i in range(5):
        dataset.append(CodeState(problem_id=f"easy_{i}", complexity=0.3))
    # 5 "Medium Leetcode" (0.6 complexity)
    for i in range(5):
        dataset.append(CodeState(problem_id=f"medium_{i}", complexity=0.6))
    # 3 "Hard Research" (0.9 complexity)
    for i in range(3):
        dataset.append(CodeState(problem_id=f"hard_{i}", complexity=0.9))

    console.print(f"Dataset size: {len(dataset)} cases.")

    # Run Simulation
    # n_runs = 20 allows us to estimate pass@k and pass^k for k up to 20.
    n_runs = 20
    console.print(f"Running Monte Carlo Simulation (n_runs={n_runs} per case)...")

    runner = MonteCarloRunner(
        graph=graph,
        dataset=dataset,
        n_runs=n_runs,
        success_criteria=lambda s: s.is_correct,
    )

    results = await runner.run_simulation()

    # Calculate Metrics
    metrics = calculate_metrics(results)

    # Display Results
    print_metrics_table(metrics)


def print_metrics_table(metrics: dict[str, Any]):
    # Summary Table
    summary = Table(title="Overall Simulation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="magenta")

    summary.add_row("Total Cases", str(metrics["total_cases"]))
    summary.add_row("Total Runs", str(metrics["total_runs"]))
    summary.add_row("Global Accuracy (pass@1)", f"{metrics['accuracy']:.2%}")
    summary.add_row(
        "Strict Consistency (all runs pass)",
        f"{metrics['consistency']:.2%}",
    )
    summary.add_row(
        "Global Reliability (at least one pass)",
        f"{metrics['reliability']:.2%}",
    )

    console.print(summary)

    # Pass@k and Pass^k Table
    k_table = Table(title="Stochastic Estimates (Unbiased Estimators)")
    k_table.add_column("k", justify="center")
    k_table.add_column("Reliability (pass@k)", style="green")
    k_table.add_column("Consistency (pass^k)", style="yellow")
    k_table.add_column("Description", style="dim")

    pass_at_k = metrics.get("pass_at_k", {})
    pass_pow_k = metrics.get("pass_pow_k", {})

    # Sort keys to ensure order
    ks = sorted([int(k.split("@")[1]) for k in pass_at_k.keys()])

    for k in ks:
        p_at = pass_at_k.get(f"pass@{k}", 0)
        p_pow = pass_pow_k.get(f"pass^{k}", 0)

        desc = ""
        if k == 1:
            desc = "Standard accuracy"
        elif k == 5:
            desc = "Best of 5"

        k_table.add_row(str(k), f"{p_at:.2%}", f"{p_pow:.2%}", desc)

    console.print(k_table)
    console.print("\n[dim]* pass@k: Probability of ≥1 success in k tries.[/dim]")
    console.print("[dim]* pass^k: Probability of k successes in k tries.[/dim]")


if __name__ == "__main__":
    asyncio.run(run_demo())
