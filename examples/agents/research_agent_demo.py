import asyncio
import json
import re

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from markov_agent import ADKConfig, BaseState, Edge, Graph, ProbabilisticNode

console = Console()

# --- 1. Define the State ---


class ResearchState(BaseState):
    """Represents the full context of a research session."""

    topic: str
    plan: list[str] = Field(default_factory=list)
    research_notes: str = ""
    draft: str = ""
    critique: str = ""
    score: int = 0
    iteration: int = 0

    def update_notes(self, new_notes: str) -> "ResearchState":
        updated_notes = (self.research_notes + "\n\n" + new_notes).strip()
        return self.update(research_notes=updated_notes)


# --- 2. Define Output Schemas ---


class PlanOutput(BaseModel):
    questions: list[str]


class ResearchOutput(BaseModel):
    notes: str


class DraftOutput(BaseModel):
    content: str


class CritiqueOutput(BaseModel):
    feedback: str
    score: int = Field(description="Score from 0 to 10")


# --- 3. Mock Responder Logic ---


class ResearchMockResponder:
    """Simulates LLM behavior for research tasks."""

    def __call__(self, prompt: str) -> str:
        if "Plan research" in prompt:
            return json.dumps(
                {
                    "questions": [
                        "History of topic?",
                        "Current state?",
                        "Future outlook?",
                    ]
                },
            )
        if "Research these questions" in prompt:
            return json.dumps(
                {
                    "notes": "Found significant data indicating growth in the sector. Users prefer modular systems.",
                },
            )
        if "Write a comprehensive article" in prompt:
            return json.dumps(
                {
                    "content": """Title: The Future of X.

Section 1: History...
Section 2: Growth...""",
                },
            )
        if "Review the following draft" in prompt:
            # Simulate improvement over iterations
            match = re.search(r"Iteration: (\d+)", prompt)
            iteration = int(match.group(1)) if match else 0

            if iteration < 1:
                return json.dumps(
                    {"feedback": "Too shallow. Needs more data.", "score": 5},
                )
            return json.dumps(
                {"feedback": "Excellent work. Comprehensive.", "score": 9},
            )
        return "{}"


# --- 4. Build the Agent Topology ---


def build_research_graph(use_local: bool = False) -> Graph:
    # Configuration
    if use_local:
        config = ADKConfig(
            model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
            api_base="http://192.168.1.213:8080/v1",
            api_key="no-key",
            use_litellm=True,
        )
    else:
        config = ADKConfig(model_name="gemini-3-flash-preview")

    mock = ResearchMockResponder()

    # Node 1: Planner
    planner = ProbabilisticNode(
        name="planner",
        adk_config=config,
        prompt_template="Plan research for: {{ topic }}. Output JSON.",
        output_schema=PlanOutput,
        state_updater=lambda s, r: s.update(plan=r.questions),
        mock_responder=mock,
        state_type=ResearchState,
    )

    # Node 2: Researcher
    researcher = ProbabilisticNode(
        name="researcher",
        adk_config=config,
        prompt_template=(
            "Research these questions: {{ plan }}. Previous notes: {{ research_notes }}. Output JSON."
        ),
        output_schema=ResearchOutput,
        state_updater=lambda s, r: s.update_notes(r.notes),
        mock_responder=mock,
        state_type=ResearchState,
    )

    # Node 3: Writer
    writer = ProbabilisticNode(
        name="writer",
        adk_config=config,
        prompt_template=(
            "Write a comprehensive article on {{ topic }} using these notes: {{ research_notes }}. Output JSON."
        ),
        output_schema=DraftOutput,
        state_updater=lambda s, r: s.update(draft=r.content),
        mock_responder=mock,
        state_type=ResearchState,
    )

    # Node 4: Critic
    critic = ProbabilisticNode(
        name="critic",
        adk_config=config,
        prompt_template=(
            "Review the following draft: {{ draft }}. Iteration: {{ iteration }}. Output JSON."
        ),
        output_schema=CritiqueOutput,
        state_updater=lambda s, r: s.update(
            critique=r.feedback,
            score=r.score,
            iteration=s.iteration + 1,
        ),
        mock_responder=mock,
        state_type=ResearchState,
    )

    # Edges & Routing Logic
    edges = [
        Edge(source="planner", target_func=lambda s: "researcher"),
        Edge(source="researcher", target_func=lambda s: "writer"),
        Edge(source="writer", target_func=lambda s: "critic"),
        Edge(
            source="critic",
            target_func=lambda s: (
                "researcher" if s.score < 8 and s.iteration < 3 else None
            ),
        ),
    ]

    return Graph(
        name="research_graph",
        nodes={
            "planner": planner,
            "researcher": researcher,
            "writer": writer,
            "critic": critic,
        },
        edges=edges,
        entry_point="planner",
        state_type=ResearchState,
    )


# --- 5. Main Execution ---


async def main():
    console.print("[bold green]Starting Research Agent Demo...[/bold green]")

    graph = build_research_graph(use_local=False)
    initial_state = ResearchState(topic="The impact of AI on Open Source")

    final_state = await graph.run(initial_state)

    console.print("\n[bold]--- Final Result ---[/bold]")
    console.print(f"Topic: [cyan]{final_state.topic}[/cyan]")
    console.print(f"Iterations: [magenta]{final_state.iteration}[/magenta]")
    console.print(f"Final Score: [yellow]{final_state.score}[/yellow]")
    console.print(Panel(final_state.draft, title="Draft Preview"))


if __name__ == "__main__":
    asyncio.run(main())
