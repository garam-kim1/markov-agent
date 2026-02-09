import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from markov_agent.containers.nested import NestedGraphNode
from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.nodes import SearchNode
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.engine.sampler import SamplingStrategy
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

# --- 1. Define Models & State ---


class Chapter(BaseModel):
    """Represents a single chapter with research and content."""

    title: str
    description: str
    research_notes: str | None = None
    content: str | None = None
    verified: bool = False


class ReportState(BaseState):
    """The global state of the report generation process."""

    topic: str
    chapters: list[Chapter] = Field(default_factory=list)
    current_chapter_index: int = 0
    final_report: str = ""


# --- 2. Output Schemas ---


class PlanOutput(BaseModel):
    chapters: list[Chapter]


class ResearchOutput(BaseModel):
    notes: str


class ChapterContentOutput(BaseModel):
    content: str


class VerificationOutput(BaseModel):
    verified: bool
    feedback: str | None = None


# --- 3. Custom Nodes for the Complex Structure ---


class DeepWriterNode(ProbabilisticNode):
    """A specialized writer that uses System 2 reasoning."""

    async def execute(self, state: ReportState) -> ReportState:
        return await self.deep(state)


# --- 4. State Updaters ---


def update_plan(state: ReportState, result: PlanOutput) -> ReportState:
    return state.update(chapters=result.chapters)


def update_research(state: ReportState, result: str | ResearchOutput) -> ReportState:
    idx = state.current_chapter_index
    chapters = list(state.chapters)
    notes = result.notes if isinstance(result, ResearchOutput) else str(result)
    if idx < len(chapters):
        chapters[idx].research_notes = notes
    return state.update(chapters=chapters)


def update_content(state: ReportState, result: ChapterContentOutput) -> ReportState:
    idx = state.current_chapter_index
    chapters = list(state.chapters)
    if idx < len(chapters):
        chapters[idx].content = result.content
    return state.update(chapters=chapters)


def update_verification(state: ReportState, result: VerificationOutput) -> ReportState:
    idx = state.current_chapter_index
    chapters = list(state.chapters)
    if idx < len(chapters):
        chapters[idx].verified = result.verified
        # Only increment index if verified
        new_idx = idx + 1 if result.verified else idx
        return state.update(chapters=chapters, current_chapter_index=new_idx)
    return state


# --- 5. Main Logic ---


async def main():
    console = Console()
    
    # Allow overriding via environment variables for local LLM support
    model_name = os.environ.get("MODEL_NAME", "gemini-3-flash-preview")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    api_base = os.environ.get("API_BASE")

    if not api_key and not api_base:
        console.print("[bold red]Error: Neither GEMINI_API_KEY nor API_BASE set.[/bold red]")
        return

    config = ADKConfig(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.7,
        enable_logging=True,
    )

    # --- TOPOLOGY 1: Chapter Workflow (The Nested Graph) ---

    researcher = SearchNode(
        name="researcher",
        adk_config=config,
        prompt_template="""
        Search for information regarding: {{ topic }}
        Specifically for the chapter: {{ chapters[current_chapter_index].title }}
        Description: {{ chapters[current_chapter_index].description }}

        Synthesize the search results into detailed research notes.
        """,
        state_updater=update_research,
        state_type=ReportState,
    )

    writer = DeepWriterNode(
        name="writer",
        adk_config=config,
        prompt_template="""
        Write the content for the chapter: {{ chapters[current_chapter_index].title }}
        Topic: {{ topic }}
        Research Notes: {{ chapters[current_chapter_index].research_notes }}

        Write professional Markdown content.
        Your output MUST match: {"content": "string"}
        """,
        output_schema=ChapterContentOutput,
        state_updater=update_content,
        samples=2,
        sampling_strategy=SamplingStrategy.DIVERSE,
        state_type=ReportState,
    )

    verifier = ProbabilisticNode(
        name="verifier",
        adk_config=config,
        prompt_template="""
        Review the following content for accuracy and style.
        Content: {{ chapters[current_chapter_index].content }}

        If it meets high standards, set verified to true.
        Your output MUST match: {"verified": boolean, "feedback": "string"}
        """,
        output_schema=VerificationOutput,
        state_updater=update_verification,
        state_type=ReportState,
    )

    chapter_edges = [
        Edge(source="researcher", target_func=lambda _: "writer"),
        Edge(source="writer", target_func=lambda _: "verifier"),
        Edge(
            source="verifier",
            target_func=lambda s: (
                "researcher"
                if not (
                    # Safety check: ensure current_chapter_index is within bounds
                    s.get("chapters", [])[s.get("current_chapter_index", 0)].get(
                        "verified",
                        False,
                    )
                    if (
                        isinstance(s, dict)
                        and s.get("current_chapter_index", 0)
                        < len(s.get("chapters", []))
                    )
                    else (
                        s.chapters[s.current_chapter_index].verified
                        if s.current_chapter_index < len(s.chapters)
                        else True  # Assume verified if out of bounds to stop loop
                    )
                )
                else None
            ),
        ),
    ]

    chapter_workflow_graph = Graph(
        name="chapter_workflow",
        nodes={"researcher": researcher, "writer": writer, "verifier": verifier},
        edges=chapter_edges,
        entry_point="researcher",
        max_steps=10,
        state_type=ReportState,
    )

    # --- TOPOLOGY 2: Orchestrator (The Main Graph) ---

    planner = ProbabilisticNode(
        name="planner",
        adk_config=config,
        prompt_template="""
        Create a 3-chapter plan for a report on: {{ topic }}
        Your output MUST match: {"chapters": [{"title": "string", "description": "string"}]}
        """,
        output_schema=PlanOutput,
        state_updater=update_plan,
        state_type=ReportState,
    )

    chapter_agent = NestedGraphNode(
        name="chapter_agent",
        graph=chapter_workflow_graph,
        state_type=ReportState,
    )

    def route_main(state: Any) -> str | None:
        # State might be dict or ReportState
        chapters = (
            state.get("chapters", []) if isinstance(state, dict) else state.chapters
        )
        idx = (
            state.get("current_chapter_index", 0)
            if isinstance(state, dict)
            else state.current_chapter_index
        )
        if idx < len(chapters):
            return "chapter_agent"
        return None

    main_edges = [
        Edge(source="planner", target_func=lambda _: "chapter_agent"),
        Edge(source="chapter_agent", target_func=route_main),
    ]

    orchestrator = Graph(
        name="report_orchestrator",
        nodes={"planner": planner, "chapter_agent": chapter_agent},
        edges=main_edges,
        entry_point="planner",
        max_steps=15,
        state_type=ReportState,
    )

    # --- Execution ---

    topic = "The Convergence of Symbolic Logic and Neural Probabilities"
    initial_state = ReportState(topic=topic)

    console.print(
        Panel(
            f"[bold green]Generating Deep Hierarchical Report[/bold green]\nTopic: [cyan]{topic}[/cyan]"
        )
    )

    try:
        final_state_data = await orchestrator.run(initial_state)

        # Ensure we have a dict to work with for assembly if it's not a model
        state_dict = (
            final_state_data
            if isinstance(final_state_data, dict)
            else final_state_data.model_dump()
        )

        # Assemble Final Report
        report_md = f"# {state_dict.get('topic')}\n\n"
        for c in state_dict.get("chapters", []):
            title = c.get("title") if isinstance(c, dict) else c.title
            content = c.get("content") if isinstance(c, dict) else c.content
            report_md += f"## {title}\n\n{content or 'No content generated.'}\n\n"

        console.print("\n[bold blue]=== FINAL REPORT ===[/bold blue]\n")
        console.print(Markdown(report_md))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    asyncio.run(main())
