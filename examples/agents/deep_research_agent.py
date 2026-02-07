import asyncio

from pydantic import BaseModel, Field

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

# --- 1. Define the State (The "Board") ---


class ResearchState(BaseState):
    """Represents the full context of a research session."""

    topic: str
    plan: list[str] = Field(default_factory=list)
    research_notes: str = ""
    draft: str = ""
    critique: str = ""
    score: int = 0
    iteration: int = 0


# --- 2. Define Output Schemas (The "Moves") ---


class PlanOutput(BaseModel):
    questions: list[str]


class ResearchOutput(BaseModel):
    notes: str


class DraftOutput(BaseModel):
    content: str


class CritiqueOutput(BaseModel):
    feedback: str
    score: int = Field(description="Score from 0 to 10")


# --- 3. Define State Updaters (The "Physics") ---


def update_plan(state: ResearchState, result: PlanOutput) -> ResearchState:
    return state.update(plan=result.questions)


def update_research(state: ResearchState, result: ResearchOutput) -> ResearchState:
    # Append new notes to existing ones
    new_notes = state.research_notes + "\n\n" + result.notes
    return state.update(research_notes=new_notes.strip())


def update_draft(state: ResearchState, result: DraftOutput) -> ResearchState:
    return state.update(draft=result.content)


def update_critique(state: ResearchState, result: CritiqueOutput) -> ResearchState:
    return state.update(
        critique=result.feedback,
        score=result.score,
        iteration=state.iteration + 1,
    )


# --- 4. Mock Logic (The "Simulation") ---
# Since we might not have a real LLM connected, we simulate behavior.


def mock_llm_router(prompt: str) -> str:
    """A fake LLM that responds based on the prompt content."""
    import json
    import re

    if "Plan research" in prompt:
        return json.dumps(
            {"questions": ["History of topic?", "Current state?", "Future outlook?"]},
        )
    if "Research these questions" in prompt:
        return json.dumps(
            {
                "notes": (
                    "Found significant data indicating growth in the sector. "
                    "Users prefer modular systems."
                ),
            },
        )
    if "Write a comprehensive article" in prompt:
        return json.dumps(
            {
                "content": (
                    "Title: The Future of X.\n\nSection 1: History...\n"
                    "Section 2: Growth..."
                ),
            },
        )
    if "Review the following draft" in prompt:
        # Simulate improvement over iterations
        match = re.search(r"Iteration: (\d+)", prompt)
        iteration = int(match.group(1)) if match else 0

        if iteration < 2:
            return json.dumps({"feedback": "Too shallow. Needs more data.", "score": 5})
        return json.dumps(
            {"feedback": "Excellent work. Comprehensive.", "score": 9},
        )
    return "{}"


# --- 5. Build the Topology (The "Game Loop") ---


async def main():
    # Configuration
    config = ADKConfig(model_name="gemini-3-flash-preview")

    # Node 1: Planner
    planner = ProbabilisticNode(
        name="planner",
        adk_config=config,
        prompt_template="Plan research for: {{ topic }}. Output JSON.",
        output_schema=PlanOutput,
        state_updater=update_plan,
        mock_responder=mock_llm_router,
        state_type=ResearchState,
    )

    # Researcher Node
    researcher = ProbabilisticNode(
        name="researcher",
        adk_config=config,
        prompt_template=(
            "Research these questions: {{ plan }}. Previous notes: {{ research_notes }}. "
            "Output JSON."
        ),
        output_schema=ResearchOutput,
        state_updater=update_research,
        mock_responder=mock_llm_router,
        state_type=ResearchState,
    )

    # Writer Node
    writer = ProbabilisticNode(
        name="writer",
        adk_config=config,
        prompt_template=(
            "Write a comprehensive article on {{ topic }} using these notes: "
            "{{ research_notes }}. Output JSON."
        ),
        output_schema=DraftOutput,
        state_updater=update_draft,
        mock_responder=mock_llm_router,
        state_type=ResearchState,
    )

    # Critic Node
    critic = ProbabilisticNode(
        name="critic",
        adk_config=config,
        prompt_template=(
            "Review the following draft: {{ draft }}. Iteration: {{ iteration }}. Output JSON."
        ),
        output_schema=CritiqueOutput,
        state_updater=update_critique,
        mock_responder=mock_llm_router,
        state_type=ResearchState,
    )

    # Edges & Routing Logic
    edges = [
        # Linear flow: Planner -> Researcher -> Writer -> Critic
        Edge(source="planner", target_func=lambda s: "researcher"),
        Edge(source="researcher", target_func=lambda s: "writer"),
        Edge(source="writer", target_func=lambda s: "critic"),
        # Conditional flow from Critic
        Edge(
            source="critic",
            target_func=lambda s: (
                "researcher"  # Loop back to research more
                if s.score < 8 and s.iteration < 5
                else None  # End if good enough or max iterations
            ),
        ),
    ]

    # Graph
    graph = Graph(
        name="deep_research_graph",
        nodes={
            "planner": planner,
            "researcher": researcher,
            "writer": writer,
            "critic": critic,
        },
        edges=edges,
        entry_point="planner",
        max_steps=20,
        state_type=ResearchState,
    )

    # Initial State
    initial_state = ResearchState(topic="The impact of AI on Coding")

    # Run
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("[bold green]Starting Deep Research Agent Simulation...[/bold green]")
    final_state = await graph.run(initial_state)

    console.print("\n[bold]--- Final Result ---[/bold]")
    console.print(f"Topic: [cyan]{final_state.topic}[/cyan]")
    console.print(f"Iterations: [magenta]{final_state.iteration}[/magenta]")
    console.print(f"Final Score: [yellow]{final_state.score}[/yellow]")
    console.print(Panel(final_state.draft, title="Draft Preview"))


if __name__ == "__main__":
    asyncio.run(main())
