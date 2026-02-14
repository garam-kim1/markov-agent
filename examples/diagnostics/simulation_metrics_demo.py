import asyncio
import random

from rich.console import Console
from rich.table import Table

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

"""
Simulation & Reliability Metrics Demo

This script demonstrates how to evaluate an agent's performance using Monte Carlo simulations.
It covers:
1. pass@1 (Standard Accuracy)
2. pass@k (Reliability: Probability of at least one success in k attempts)
3. pass^k (Consistency: Probability of k successes in k attempts)

It provides two examples:
- Math Solver: Comparing baseline (k=1) vs runtime majority vote (k=5).
- Code Generator: Stochastic simulation across varying problem complexities.
"""

console = Console()

# --- 1. Math Domain Example ---


class MathState(BaseState):
    question: str
    expected_answer: int
    solution: int | None = None


def math_mock_responder(prompt: str) -> dict:
    # Simulate a model that is 70% accurate
    is_correct = random.random() < 0.7
    if "12 + 25" in prompt:
        return {"answer": 37 if is_correct else 40}
    return {"answer": 0}


def build_math_agent(samples: int = 1) -> Graph:
    solver = ProbabilisticNode(
        name="solver",
        adk_config=ADKConfig(model_name="mock-math"),
        prompt_template="Solve: {{ question }}",
        mock_responder=math_mock_responder,
        state_updater=lambda s, r: s.update(solution=r.get("answer")),
        samples=samples,
        state_type=MathState,
    )
    return Graph(
        name="math_agent",
        nodes={"solver": solver},
        edges=[Edge(source="solver", target_func=lambda s: None)],
        entry_point="solver",
    )


# --- 2. Code Domain Example ---


class CodeState(BaseState):
    problem_id: str
    complexity: float  # 0.0 to 1.0
    is_correct: bool = False


def code_mock_responder(prompt: str) -> dict:
    # Probability of success = 1.0 - complexity
    complexity = 0.5
    if "COMPLEXITY:" in prompt:
        complexity = float(prompt.split("COMPLEXITY:")[1].strip())

    is_success = random.random() < (1.0 - complexity)
    return {"valid": is_success}


def build_code_agent() -> Graph:
    coder = ProbabilisticNode(
        name="coder",
        adk_config=ADKConfig(model_name="mock-coder"),
        prompt_template="Solve {{ problem_id }}. COMPLEXITY: {{ complexity }}",
        mock_responder=code_mock_responder,
        state_updater=lambda s, r: s.update(is_correct=r.get("valid")),
        state_type=CodeState,
    )
    return Graph(
        name="code_agent",
        nodes={"coder": coder},
        edges=[Edge(source="coder", target_func=lambda s: None)],
        entry_point="coder",
    )


# --- 3. Run Simulations ---


async def run_math_sim():
    console.print("\n[bold cyan]--- Math Reliability Simulation ---[/bold cyan]")
    dataset = [
        MathState(question="What is 12 + 25?", expected_answer=37),
    ]
    agent = build_math_agent(samples=1)
    runner = MonteCarloRunner(
        graph=agent,
        dataset=dataset,
        n_runs=20,
        success_criteria=lambda s: s.solution == s.expected_answer,
    )
    results = await runner.run_simulation()
    metrics = calculate_metrics(results)
    print_metrics_table("Math Solver Performance", metrics)


async def run_code_sim():
    console.print("\n[bold cyan]--- Code Complexity Simulation ---[/bold cyan]")
    dataset = [
        CodeState(problem_id="easy_task", complexity=0.2),
        CodeState(problem_id="hard_task", complexity=0.8),
    ]
    agent = build_code_agent()
    runner = MonteCarloRunner(
        graph=agent,
        dataset=dataset,
        n_runs=20,
        success_criteria=lambda s: s.is_correct,
    )
    results = await runner.run_simulation()
    metrics = calculate_metrics(results)
    print_metrics_table("Code Generator Performance", metrics)


def print_metrics_table(title: str, metrics: dict):
    table = Table(title=title)
    table.add_column("k", justify="center")
    table.add_column("Reliability (pass@k)", style="green")
    table.add_column("Consistency (pass^k)", style="yellow")

    pass_at_k = metrics.get("pass_at_k", {})
    pass_pow_k = metrics.get("pass_pow_k", {})

    for k_str in sorted(pass_at_k.keys(), key=lambda x: int(x.split("@")[1])):
        k = k_str.split("@")[1]
        table.add_row(
            k, f"{pass_at_k[f'pass@{k}']:.2%}", f"{pass_pow_k[f'pass^{k}']:.2%}"
        )

    console.print(table)


async def main():
    await run_math_sim()
    await run_code_sim()


if __name__ == "__main__":
    asyncio.run(main())
