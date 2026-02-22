import asyncio

from rich.console import Console

from markov_agent import ADKConfig, BaseState, Edge, Graph, ProbabilisticNode
from markov_agent.engine.eig import EntropyCheck
from markov_agent.engine.mcts import MCTSNode
from markov_agent.governance.cost import CostGovernor

console = Console()


class ResearchState(BaseState):
    query: str = ""
    entropy_score: float = 0.0
    next_step: str = ""
    research_notes: str = ""
    final_answer: str = ""


async def run_demo():
    # 1. Setup Governance
    cheap_cfg = ADKConfig(model_name="gemini-3-flash-preview", temperature=0.1)
    reasoning_cfg = ADKConfig(model_name="gemini-3-flash-preview", temperature=0.7)

    governor = CostGovernor(
        cheap_config=cheap_cfg, reasoning_config=reasoning_cfg, cost_budget=0.5
    )

    # 2. Define Nodes

    # Entropy Check Node (Information Gain Router)
    router_node = EntropyCheck(
        name="router",
        adk_config=cheap_cfg,
        threshold=0.6,
        clarification_node="ask_user",
        execution_node="mcts_search",
        state_type=ResearchState,
    )

    # MCTS Node (System 2 Search)
    mcts_node = MCTSNode(
        name="mcts_search",
        adk_config=reasoning_cfg,
        max_rollouts=5,
        expansion_k=2,
        state_type=ResearchState,
    )

    # Simple Execution Node
    clarify_node = ProbabilisticNode(
        name="ask_user",
        adk_config=cheap_cfg,
        prompt_template="The query '{query}' is ambiguous. Please provide more context.",
        state_type=ResearchState,
    )

    # 3. Define Graph
    nodes = {"router": router_node, "mcts_search": mcts_node, "ask_user": clarify_node}

    edges = [
        Edge("router", lambda s: s.next_step),
        Edge("ask_user", lambda s: "mcts_search"),  # Loop back after clarification
    ]

    graph = Graph(
        name="System2Graph",
        nodes=nodes,
        edges=edges,
        entry_point="router",
        state_type=ResearchState,
    )

    # 4. Execute
    initial_state = ResearchState(query="Quantum computing impact on cryptography")

    console.print("[bold blue]Starting System 2 & Governance Demo[/bold blue]")

    # Route via governor (Simulated complexity check)
    selected_cfg = governor.route_request(complexity_score=0.8)
    console.print(f"Governor selected model: [green]{selected_cfg.model_name}[/green]")

    final_state = await graph.run(initial_state)

    console.print("\n[bold green]Final State:[/bold green]")
    console.print(final_state.model_dump())


if __name__ == "__main__":
    # Ensure Mocking if no API key is set (Markov Agent convention)
    # For demo we assume environment is set up
    asyncio.run(run_demo())
