import asyncio

from pydantic import BaseModel, Field
from rich.console import Console

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, RetryPolicy
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph

# Setup Rich Console
console = Console()

# --- 1. Define State ---


class ResearchStep(BaseModel):
    id: int
    question: str
    answer: str | None = None
    status: str = "pending"  # pending, completed, failed


class ResearchState(BaseState):
    query: str
    plan: list[ResearchStep] = Field(default_factory=list)
    final_report: str | None = None
    current_step_index: int = 0

    def get_current_step(self) -> ResearchStep | None:
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    def update_step_answer(self, answer: str) -> "ResearchState":
        new_state = self.model_copy(deep=True)
        new_state.plan[new_state.current_step_index].answer = answer
        new_state.plan[new_state.current_step_index].status = "completed"
        new_state.current_step_index += 1
        return new_state

    def set_plan(self, steps: list[str]) -> "ResearchState":
        new_state = self.model_copy(deep=True)
        new_state.plan = [ResearchStep(id=i, question=q) for i, q in enumerate(steps)]
        return new_state

    def set_report(self, report: str) -> "ResearchState":
        new_state = self.model_copy(deep=True)
        new_state.final_report = report
        return new_state


# --- 2. Configuration (The User's Local LLM) ---

LOCAL_LLM_CONFIG = ADKConfig(
    model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
    api_base="http://192.168.1.213:8080/v1",
    api_key="no-key",
    temperature=0.7,
)

RETRY_POLICY = RetryPolicy(max_attempts=3)

# --- 3. Define Output Schemas ---


class PlanOutput(BaseModel):
    steps: list[str]


class AnswerOutput(BaseModel):
    answer: str
    confidence: float


class ReportOutput(BaseModel):
    summary: str


# --- 4. Define Nodes ---

# Node 1: Planner
planner = ProbabilisticNode(
    name="planner",
    adk_config=LOCAL_LLM_CONFIG,
    prompt_template="""
    You are a research planner.
    Break down the user's query into 3 distinct, logical research questions.
    
    User Query: {query}
    
    Return JSON format with a 'steps' list of strings.
    """,
    output_schema=PlanOutput,
    retry_policy=RETRY_POLICY,
    state_updater=lambda s, r: s.set_plan(r.steps),
)

# Node 2: Executor
executor = ProbabilisticNode(
    name="executor",
    adk_config=LOCAL_LLM_CONFIG,
    prompt_template="""
    Answer the following question concisely based on general knowledge.
    
    Question: {plan[current_step_index].question}
    
    Return JSON with 'answer' and 'confidence' (0.0 to 1.0).
    """,
    output_schema=AnswerOutput,
    retry_policy=RETRY_POLICY,
    state_updater=lambda s, r: s.update_step_answer(r.answer),
)

# Node 3: Synthesizer
synthesizer = ProbabilisticNode(
    name="synthesizer",
    adk_config=LOCAL_LLM_CONFIG,
    prompt_template="""
    Create a final answer for the original query based on the research steps.
    
    Original Query: {query}
    
    Research Findings:
    {% for step in plan %}
    - Q: {{ step.question }}
      A: {{ step.answer }}
    {% endfor %}
    
    Return JSON with a 'summary' field.
    """,
    output_schema=ReportOutput,
    retry_policy=RETRY_POLICY,
    state_updater=lambda s, r: s.set_report(r.summary),
)

# --- 5. Topology (The Graph) ---


def router(state: ResearchState) -> str | None:
    # Logic to determine next node

    # If no plan, go to planner (Logic handled by entry point usually, but for loops:)
    if not state.plan:
        return "planner"

    # If we have a current step, execute it
    if state.get_current_step():
        return "executor"

    # If all steps done (no current step), synthesize
    if state.final_report is None:
        return "synthesizer"

    # Done
    return None


# Edges
# Note: Edge takes (source, target_func).
# target_func receives state and returns next node ID.
edge_planner = Edge(source="planner", target_func=router)
edge_executor = Edge(source="executor", target_func=router)
edge_synthesizer = Edge(source="synthesizer", target_func=lambda s: None)  # End

# Graph Construction
graph = Graph(
    entry_point="planner",
    nodes={"planner": planner, "executor": executor, "synthesizer": synthesizer},
    edges=[edge_planner, edge_executor, edge_synthesizer],
)

# --- 6. Execution ---


async def main():
    console.print("[bold green]Starting Complex Local Agent Run...[/bold green]")

    # Initial State
    initial_state = ResearchState(
        query="Explain the impact of quantum computing on cryptography."
    )

    try:
        # Run the graph
        # Since we are using a custom endpoint, this relies on the ADK wrapper updates
        # removed max_steps if it defaults to 50
        final_state = await graph.run(initial_state)

        console.print("\n[bold blue]--- Final Report ---[/bold blue]")
        console.print(final_state.final_report)
        console.print("[bold blue]--------------------[/bold blue]")

    except Exception as e:
        console.print(f"[bold red]Execution Failed:[/bold red] {e}")
        # Print stack trace for debugging if needed
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
