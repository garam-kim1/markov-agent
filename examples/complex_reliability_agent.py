import asyncio
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.simulation.metrics import calculate_metrics
from markov_agent.simulation.runner import MonteCarloRunner
from markov_agent.topology.graph import Graph

console = Console()

# -------------------------------------------------------------------------
# 1. Define the Domain (Math Problems)
# -------------------------------------------------------------------------


class MathProblem(BaseModel):
    question: str
    expected_answer: int


class MathSolution(BaseModel):
    reasoning: str
    final_answer: int


class MathState(BaseState):
    problem: MathProblem
    solution: MathSolution | None = None

    def get_prompt(self) -> str:
        return f"Solve this math problem: {self.problem.question}. Return JSON with 'reasoning' and 'final_answer'."


# -------------------------------------------------------------------------
# 2. Define the Configuration
# -------------------------------------------------------------------------

LOCAL_LLM_CONFIG = ADKConfig(
    model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
    api_base="http://192.168.1.213:8080/v1",
    api_key="dummy",
    use_litellm=True,
    temperature=0.7,  # High temperature for diversity in sampling
)

# -------------------------------------------------------------------------
# 3. Define the Selector (Runtime Verification)
# -------------------------------------------------------------------------


def math_selector(results: list[Any]) -> Any:
    """
    Selects the most common answer (Self-Consistency) or the first valid one.
    For this example, we'll use a majority vote on 'final_answer'.
    """
    valid_results = [r for r in results if isinstance(r, MathSolution)]
    if not valid_results:
        return results[0]  # Return whatever we got (maybe error or None)

    # Majority vote
    votes = {}
    for res in valid_results:
        ans = res.final_answer
        votes[ans] = votes.get(ans, 0) + 1

    best_ans = max(votes, key=votes.get)

    # Return one of the solutions that has the best answer
    for res in valid_results:
        if res.final_answer == best_ans:
            return res

    return valid_results[0]


# -------------------------------------------------------------------------
# 4. Define the Agent Topology
# -------------------------------------------------------------------------


def build_math_agent(samples: int = 1, use_selector: bool = False) -> Graph:
    # Node: Solver
    solver_node = ProbabilisticNode(
        name="solver",
        adk_config=LOCAL_LLM_CONFIG,
        prompt_template="{get_prompt}",  # Uses method on State
        output_schema=MathSolution,
        samples=samples,
        selector=math_selector if use_selector else None,
        state_type=MathState,
        state_updater=lambda state, result: state.model_copy(
            update={"solution": result}
        ),
    )

    # Simple 1-node graph
    graph = Graph(start_node="solver", state_type=MathState)
    graph.add_node(solver_node)

    # Edge: Done
    graph.add_edge("solver", lambda s: None)  # End

    return graph


# -------------------------------------------------------------------------
# 5. Define the Simulation / Experiment
# -------------------------------------------------------------------------


async def run_experiment():
    console.print("[bold blue]Running Complex Reliability Experiment[/bold blue]")

    # Dataset: 5 Simple Math Problems
    dataset = [
        MathState(problem=MathProblem(question="What is 12 + 25?", expected_answer=37)),
        MathState(
            problem=MathProblem(question="Calculate 5 * 8 - 3", expected_answer=37)
        ),
        MathState(
            problem=MathProblem(
                question="If x = 10, what is 2x + 5?", expected_answer=25
            )
        ),
        MathState(
            problem=MathProblem(
                question="What is the square root of 144?", expected_answer=12
            )
        ),
        MathState(
            problem=MathProblem(question="Solve for x: 3x = 27", expected_answer=9)
        ),
    ]

    # --- Experiment A: Baseline (k=1, n_runs=10) ---
    # We run the agent 10 times per problem to calculate pass@k metrics
    console.print(
        "\n[bold yellow]--- Experiment A: Baseline Simulation (Calculating pass@k metrics) ---[/bold yellow]"
    )
    console.print(
        "Running standard agent (k=1) 10 times per case to estimate reliability curves..."
    )

    agent_baseline = build_math_agent(samples=1, use_selector=False)

    runner = MonteCarloRunner(
        graph=agent_baseline,
        dataset=dataset,
        n_runs=5,  # 5 runs per case to save time, but sufficient for pass@5
        success_criteria=lambda s: s.solution
        and s.solution.final_answer == s.problem.expected_answer,
    )

    results = await runner.run_simulation()
    metrics = calculate_metrics(results)

    console.print("\n[bold green]Baseline Metrics:[/bold green]")
    console.print(f"Global Accuracy (pass@1): {metrics['accuracy']:.2f}")
    console.print(f"Consistency (pass^5): {metrics['consistency']:.2f}")
    console.print(f"Reliability (pass@5): {metrics['reliability']:.2f}")

    console.print(
        "\n[bold cyan]Estimated pass@k (Probability of at least 1 correct in k tries):[/bold cyan]"
    )
    for k, v in metrics["pass_at_k"].items():
        console.print(f"  {k}: {v:.4f}")

    console.print(
        "\n[bold cyan]Estimated pass^k (Probability of ALL correct in k tries):[/bold cyan]"
    )
    for k, v in metrics["pass_pow_k"].items():
        console.print(f"  {k}: {v:.4f}")

    # --- Experiment B: Runtime Reliability (k=5 with Selector) ---
    # We verify if using k=5 at runtime actually improves the single-shot performance
    console.print(
        "\n[bold yellow]--- Experiment B: Runtime Reliability (Self-Consistency with k=5) ---[/bold yellow]"
    )
    console.print("Running enhanced agent (k=5, majority vote) 1 time per case...")

    agent_enhanced = build_math_agent(samples=5, use_selector=True)

    # We treat this as a "n_runs=1" simulation, but the agent itself does 5 internal calls
    runner_enhanced = MonteCarloRunner(
        graph=agent_enhanced,
        dataset=dataset,
        n_runs=1,
        success_criteria=lambda s: s.solution
        and s.solution.final_answer == s.problem.expected_answer,
    )

    results_enhanced = await runner_enhanced.run_simulation()
    metrics_enhanced = calculate_metrics(results_enhanced)

    console.print("\n[bold green]Enhanced Metrics (Runtime k=5):[/bold green]")
    console.print(f"Accuracy: {metrics_enhanced['accuracy']:.2f}")

    # Comparison
    improvement = metrics_enhanced["accuracy"] - metrics["accuracy"]
    console.print(f"\n[bold magenta]Improvement:[/bold magenta] {improvement:+.2f}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
