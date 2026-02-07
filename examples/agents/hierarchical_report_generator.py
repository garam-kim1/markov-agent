import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

# --- 1. Define Models & State ---


class Chapter(BaseModel):
    """Represents a single chapter in the report."""

    title: str
    description: str
    content: str | None = None


class ReportState(BaseState):
    """The state of the report generation process."""

    topic: str
    chapters: list[Chapter] = Field(default_factory=list)
    current_chapter_index: int = 0
    final_report: str = ""


# --- 2. Output Schemas (For JSON Validation) ---


class PlanOutput(BaseModel):
    """Output of the leader agent (planner)."""

    chapters: list[Chapter]


class ChapterContentOutput(BaseModel):
    """Output of the sub-agent (writer) for a single chapter."""

    content: str


# --- 3. State Updaters ---


def update_plan(state: Any, result: PlanOutput) -> dict[str, Any]:
    """Updates the state with the list of planned chapters.

    Handles both ReportState objects and raw dictionaries.
    """
    chapters = [
        c.model_dump() if isinstance(c, BaseModel) else c for c in result.chapters
    ]

    if hasattr(state, "update"):
        return state.update(chapters=chapters)

    # Fallback for dict
    new_state = dict(state)
    new_state["chapters"] = chapters
    return new_state


def update_chapter_content(state: Any, result: ChapterContentOutput) -> dict[str, Any]:
    """Updates the content of the current chapter and increments the index."""
    # Robustly get current values
    if isinstance(state, BaseModel):
        chapters = [
            c.model_dump() if isinstance(c, BaseModel) else c for c in state.chapters
        ]
        idx = state.current_chapter_index
    else:
        chapters = list(state.get("chapters", []))
        idx = state.get("current_chapter_index", 0)

    if idx < len(chapters):
        # Update the chapter at the current index
        chapter_data = dict(chapters[idx])
        chapter_data["content"] = result.content
        chapters[idx] = chapter_data

    updates = {"chapters": chapters, "current_chapter_index": idx + 1}

    if hasattr(state, "update"):
        return state.update(**updates)

    new_state = dict(state)
    new_state.update(updates)
    return new_state


def finalize_report(state: ReportState) -> ReportState:
    """Concatenates all chapters into a single markdown report."""
    full_text = f"# Report: {state.topic}\n\n"
    for chapter in state.chapters:
        # Chapter might be a dict or a Chapter object
        title = (
            chapter.title
            if hasattr(chapter, "title")
            else chapter.get("title", "Unknown Title")
        )
        content = (
            chapter.content if hasattr(chapter, "content") else chapter.get("content")
        )

        full_text += f"## {title}\n\n{content or 'No content generated.'}\n\n"

    return state.update(final_report=full_text)


# --- 4. Main Agent Logic ---


async def main():
    console = Console()
    # Retrieve the API key from the environment variable GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        console.print(
            "[bold red]Error: GEMINI_API_KEY environment variable not set.[/bold red]",
        )
        console.print("Please set it with: export GEMINI_API_KEY='your-key-here'")
        return

    # Configuration for the gemini-3-flash-preview model.
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        api_key=api_key,
        temperature=0.7,
    )

    # Leader Node: Chapter Planner
    planner = ProbabilisticNode(
        name="planner",
        adk_config=config,
        prompt_template="""
        You are a lead editor and strategist. Your task is to analyze the following user input topic
        and generate a comprehensive report structure.

        Topic: {{ topic }}

        Generate exactly 3 distinct chapters. For each chapter, provide:
        1. A compelling title.
        2. A brief but clear description of what the sub-agent should write about in this chapter.

        Your output MUST be a valid JSON object matching this schema:
        {"chapters": [{"title": "string", "description": "string"}]}
        """,
        output_schema=PlanOutput,
        state_updater=update_plan,
        state_type=ReportState,
    )

    # Sub-agent Node: Chapter Writer
    writer = ProbabilisticNode(
        name="writer",
        adk_config=config,
        prompt_template="""
        You are an expert technical writer. You are writing a specific chapter for a report on: {{ topic }}

        Current Chapter Info:
        {% set current_chapter = chapters[current_chapter_index] %}
        Title: {{ current_chapter.title if current_chapter.title is defined else current_chapter['title'] }}
        Description: {{ current_chapter.description if current_chapter.description is defined else current_chapter['description'] }}

        Write the full, detailed content for this chapter in professional Markdown format.
        Include headers, bullet points, and clear explanations.

        Your output MUST be a valid JSON object matching this schema:
        {"content": "string"}
        """,
        output_schema=ChapterContentOutput,
        state_updater=update_chapter_content,
        state_type=ReportState,
    )

    # Define the Topology (The Graph)
    def route_writer(state: Any) -> str | None:
        if isinstance(state, BaseModel):
            chapters = state.chapters
            idx = state.current_chapter_index
        else:
            chapters = state.get("chapters", [])
            idx = state.get("current_chapter_index", 0)

        return "writer" if idx < len(chapters) else None

    edges = [
        Edge(source="planner", target_func=lambda s: "writer"),
        Edge(source="writer", target_func=route_writer),
    ]

    graph = Graph(
        name="hierarchical_report_generator",
        nodes={
            "planner": planner,
            "writer": writer,
        },
        edges=edges,
        entry_point="planner",
        max_steps=20,
        state_type=ReportState,
    )

    # User Input
    user_topic = "The Role of Stochastic PPUs in Modern AI Architectures"
    initial_state = ReportState(topic=user_topic)

    console.print(
        "[bold green]Starting Hierarchical Report Generation...[/bold green]",
    )
    console.print(f"Topic: [cyan]{user_topic}[/cyan]\n")

    try:
        # Run the Markov Agent Graph
        final_state = await graph.run(initial_state)

        # Assemble the final report from chapters
        final_state = finalize_report(final_state)

        console.print("\n[bold blue]=== FINAL GENERATED REPORT ===[/bold blue]\n")
        console.print(Markdown(final_state.final_report))

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        import traceback

        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
