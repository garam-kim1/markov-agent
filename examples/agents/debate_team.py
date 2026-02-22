import asyncio
import json
import random
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from markov_agent import (
    ADKConfig,
    BaseState,
    Edge,
    Graph,
    ProbabilisticNode,
    RetryPolicy,
)

# Initialize Rich Console
console = Console()

# --- 1. Define State ---


class DialogueMessage(BaseModel):
    sender: str
    content: str


class DebateState(BaseState):
    topic: str
    history: list[DialogueMessage] = Field(default_factory=list)
    round: int = 0
    status: Literal["ongoing", "consensus", "deadlock", "visionary", "pragmatist"] = (
        "ongoing"
    )
    final_summary: str | None = None

    def add_message(self, sender: str, content: str) -> "DebateState":
        new_state = self.model_copy(deep=True)
        new_state.history.append(DialogueMessage(sender=sender, content=content))
        return new_state

    def increment_round(self) -> "DebateState":
        new_state = self.model_copy(deep=True)
        new_state.round += 1
        return new_state

    def update_status(
        self,
        status: Literal["ongoing", "consensus", "deadlock", "visionary", "pragmatist"],
        summary: str | None = None,
    ) -> "DebateState":
        new_state = self.model_copy(deep=True)
        new_state.status = status
        if summary:
            new_state.final_summary = summary
        return new_state

    @property
    def last_message(self) -> str:
        if not self.history:
            return "None"
        last = self.history[-1]
        if isinstance(last, dict):
            return f"{last.get('sender')}: {last.get('content')}"
        return f"{last.sender}: {last.content}"

    def get_context(self) -> str:
        msgs = []
        for m in self.history:
            if isinstance(m, dict):
                msgs.append(f"{m.get('sender')}: {m.get('content')}")
            else:
                msgs.append(f"{m.sender}: {m.content}")
        return "\n".join(msgs)


# --- 2. Mock Logic (Simulates LLM behavior) ---


class MockDebateLLM:
    """Simulates a debate between a Visionary (Optimist) and a Pragmatist (Critic),
    moderated by an Orchestrator.
    """

    def __call__(self, prompt: str) -> str:
        # Determine which agent is being prompted based on the prompt content

        if "You are a Visionary" in prompt:
            # Visionary Logic
            if "Space Elevators" in prompt:
                responses = [
                    "We build a carbon nanotube tether anchored to a floating platform in the Pacific!",
                    "We should add a luxury hotel at the geostationary station for tourism revenue.",
                    "The counterweight can be a captured asteroid, mining it for resources!",
                ]
                return json.dumps(
                    {
                        "thought": "Thinking big...",
                        "proposal": random.choice(responses),
                    },
                )
            return json.dumps(
                {"thought": "Ideating...", "proposal": "Let's build a Dyson Sphere!"},
            )

        if "You are a Pragmatist" in prompt:
            # Pragmatist Logic
            if "nanotube" in prompt:
                return json.dumps(
                    {
                        "thought": "Checking physics...",
                        "critique": "Current material science cannot produce nanotubes of that length without defects. It will snap.",
                    },
                )
            if "luxury hotel" in prompt:
                return json.dumps(
                    {
                        "thought": "Checking budget...",
                        "critique": "The radiation shielding requirements for a hotel at that altitude make the payload mass prohibitive.",
                    },
                )
            if "asteroid" in prompt:
                return json.dumps(
                    {
                        "thought": "Checking safety...",
                        "critique": "Capturing an asteroid poses an unacceptable risk of orbital decay and impact.",
                    },
                )
            return json.dumps(
                {
                    "thought": "Being skeptical...",
                    "critique": "That sounds incredibly expensive and risky.",
                },
            )

        if "You are the Moderator" in prompt:
            # Moderator Logic
            # Check context from prompt (simplistic check for mock)
            if "round: 3" in prompt or "round: 4" in prompt or "round: 5" in prompt:
                return json.dumps(
                    {
                        "analysis": "Enough discussion.",
                        "decision": "consensus",
                        "summary": "We agreed to focus on material science R&D before construction.",
                    },
                )

            # Routing based on last speaker in history
            last_visionary = prompt.rfind("Visionary:")
            last_pragmatist = prompt.rfind("Pragmatist:")

            if last_visionary > last_pragmatist:
                # Visionary spoke last
                return json.dumps(
                    {
                        "analysis": "Visionary proposed.",
                        "decision": "pragmatist",
                        "summary": "",
                    },
                )
            if last_pragmatist > last_visionary:
                # Pragmatist spoke last
                return json.dumps(
                    {
                        "analysis": "Critique received.",
                        "decision": "visionary",
                        "summary": "",
                    },
                )
            # Neither found (start) or equal (impossible if strictly alternating lines)
            return json.dumps(
                {
                    "analysis": "Starting debate.",
                    "decision": "visionary",
                    "summary": "",
                },
            )

        return "{}"


mock_llm = MockDebateLLM()


# --- 3. Define Schemas ---


class VisionaryOutput(BaseModel):
    thought: str
    proposal: str


class PragmatistOutput(BaseModel):
    thought: str
    critique: str


class ModeratorOutput(BaseModel):
    analysis: str
    decision: Literal["visionary", "pragmatist", "consensus", "deadlock"]
    summary: str


# --- 4. Configure Agents ---

COMMON_CONFIG = ADKConfig(model_name="gemini-3-flash-preview", temperature=0.7)
RETRY = RetryPolicy(max_attempts=3)

# Agent 1: Visionary
visionary = ProbabilisticNode(
    name="visionary",
    adk_config=COMMON_CONFIG,
    prompt_template="""
    You are a Visionary. Propose innovative solutions for the topic: {{ topic }}.

    Current Context:
    {{ state.get_context() }}

    Respond with a JSON object containing 'thought' and 'proposal'.
    """,
    output_schema=VisionaryOutput,
    retry_policy=RETRY,
    mock_responder=mock_llm,
    state_updater=lambda s, r: s.add_message("Visionary", r.proposal),
    state_type=DebateState,
)

# Agent 2: Pragmatist
pragmatist = ProbabilisticNode(
    name="pragmatist",
    adk_config=COMMON_CONFIG,
    prompt_template="""
    You are a Pragmatist. Critique the Visionary's ideas based on feasibility and cost.

    Current Context:
    {{ state.get_context() }}

    Respond with a JSON object containing 'thought' and 'critique'.
    """,
    output_schema=PragmatistOutput,
    retry_policy=RETRY,
    mock_responder=mock_llm,
    state_updater=lambda s, r: s.add_message("Pragmatist", r.critique),
    state_type=DebateState,
)

# Agent 3: Moderator (The Orchestra Conductor)
moderator = ProbabilisticNode(
    name="moderator",
    adk_config=COMMON_CONFIG,
    prompt_template="""
    You are the Moderator. Analyze the debate.
    Topic: {{ topic }}
    Round: {{ round }}

    History:
    {{ state.get_context() }}

    Decide who speaks next: 'visionary' or 'pragmatist'.
    If they have reached agreement or if it's round 3, decide 'consensus'.

    Respond with JSON: 'analysis', 'decision', 'summary' (only if consensus).
    """,
    output_schema=ModeratorOutput,
    retry_policy=RETRY,
    mock_responder=mock_llm,
    state_updater=lambda s, r: s.update_status(r.decision, r.summary).increment_round(),
    state_type=DebateState,
)

# --- 5. Topology ---


def moderator_router(state: DebateState) -> str | None:
    # The moderator agent updates state.status to 'visionary', 'pragmatist', or 'consensus'
    if state.status == "consensus" or state.status == "deadlock":
        return None  # Stop
    if state.status == "visionary":
        return "visionary"
    if state.status == "pragmatist":
        return "pragmatist"
    return None


# Graph Construction
# Flow: Start -> Moderator (Decides who starts) -> Agent -> Moderator -> ...
edges = [
    Edge(source="visionary", target_func=lambda s: "moderator"),
    Edge(source="pragmatist", target_func=lambda s: "moderator"),
    Edge(source="moderator", target_func=moderator_router),
]

debate_graph = Graph(
    name="DebateTeam",
    nodes={"visionary": visionary, "pragmatist": pragmatist, "moderator": moderator},
    edges=edges,
    entry_point="moderator",
    state_type=DebateState,
    max_steps=10,
)

# --- 6. Execution ---


async def main():
    console.print(
        Panel.fit(
            "[bold magenta]Complex Agent Example: Deep Dialogue Debate[/bold magenta]",
            subtitle="Orchestrated by Markov Graph",
        ),
    )

    initial_state = DebateState(topic="Feasibility of Space Elevators")

    console.print(f"[bold]Topic:[/bold] {initial_state.topic}\n")

    # We attach a simple observer to print messages as they happen
    # In a real app, we'd use the event bus. For this script, we just inspect final state
    # or rely on the console logs inside Graph (if enabled).
    # Let's verify by printing the final history.

    try:
        final_state = await debate_graph.run(initial_state)

        console.print("\n[bold green]--- Debate Transcript ---[/bold green]")
        for msg in final_state.history:
            sender = msg.get("sender") if isinstance(msg, dict) else msg.sender
            content = msg.get("content") if isinstance(msg, dict) else msg.content

            color = "cyan" if sender == "Visionary" else "yellow"
            console.print(f"[{color}][bold]{sender}:[/bold] {content}[/{color}]")

        console.print("\n[bold blue]--- Moderator Summary ---[/bold blue]")
        console.print(final_state.final_summary)
        console.print(f"[dim]Total Rounds: {final_state.round}[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
