from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.events import Event, EventActions
from pydantic import ConfigDict, Field

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.node import BaseNode

try:
    from rich.console import Console
    from rich.panel import Panel as RichPanel

    console = Console()

    def panel(x: Any, title: str | None = None) -> Any:
        return RichPanel(x, title=title)
except ImportError:

    class Console:
        def log(self, *args: Any, **kwargs: Any) -> None:
            pass

        def print(self, *args: Any, **kwargs: Any) -> None:
            pass

    console = Console()

    def panel(x: Any, title: str | None = None) -> Any:  # noqa: ARG001
        return x


StateT = TypeVar("StateT", bound=BaseState)


class Graph(BaseAgent):
    """The execution engine acting as a finite state machine, wrapped as an ADK Agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Re-declare fields compatible with Pydantic/ADK
    nodes: dict[str, BaseNode] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    entry_point: str = ""
    max_steps: int = 50
    state_type: type[StateT] | None = None
    input_key: str = "input_text"

    def __init__(
        self,
        name: str,
        nodes: dict[str, BaseNode],
        edges: list[Edge],
        entry_point: str,
        state_type: type[StateT] | None = None,
        input_key: str = "input_text",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.state_type = state_type
        self.input_key = input_key

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Execute the graph topology within the ADK runtime."""
        # 0. Handle Input Injection (ADK API Server support)
        if ctx.user_content and ctx.user_content.parts:
            # Extract text from the first part
            input_text = ""
            for part in ctx.user_content.parts:
                if part.text:
                    input_text += part.text

            if input_text:
                console.log(
                    f"Injecting input into state['{self.input_key}']: "
                    f"{input_text[:50]}...",
                )
                ctx.session.state[self.input_key] = input_text

        current_node_id = self.entry_point
        steps = 0

        console.log(
            f"[bold green]Starting Graph Execution[/bold green] at {self.entry_point}",
        )

        # We need to ensure we are working with the latest state from the session
        # ctx.session.state is a MutableMapping (dict-like)

        while steps < self.max_steps:
            if current_node_id not in self.nodes:
                yield Event(author=self.name, actions=EventActions(escalate=True))
                break

            current_node = self.nodes[current_node_id]

            console.log(
                panel(
                    f"Executing Node: [cyan]{current_node_id}[/cyan]",
                    title="Step info",
                ),
            )

            # Execute Node
            # We call the node's ADK implementation directly
            # This allows the node to read/write to ctx.session.state
            async for event in current_node._run_async_impl(ctx):
                yield event

            # Transition Logic
            # We construct the typed state object for the router if possible

            state_obj: Any = ctx.session.state
            if self.state_type:
                try:
                    state_obj = self.state_type.model_validate(ctx.session.state)
                except Exception:
                    # Best effort construct
                    state_obj = self.state_type.construct(**ctx.session.state)
            else:
                # Fallback Proxy
                class StateProxy:
                    def __init__(self, data: dict[str, Any]) -> None:
                        self.__dict__ = data

                    def __getattr__(self, name: str) -> Any:
                        return self.__dict__.get(name)

                state_obj = StateProxy(ctx.session.state)

            # Find next node
            next_node_id = None
            chosen_prob = 1.0

            for edge in self.edges:
                if edge.source == current_node_id:
                    # Capture both node and probability
                    next_node_id, chosen_prob = edge.route(state_obj)

                    if next_node_id:
                        console.log(
                            f"Transition: {current_node_id} -> {next_node_id} "
                            f"(p={chosen_prob:.2f})",
                        )

                        # Sync back to session state if it was a Markov State
                        if hasattr(state_obj, "meta"):
                            ctx.session.state["meta"] = state_obj.meta
                    break

            if next_node_id is None:
                console.log(
                    f"[bold yellow]Terminal node reached:[/bold yellow] "
                    f"{current_node_id}",
                )
                break

            current_node_id = next_node_id
            steps += 1

        if steps >= self.max_steps:
            console.log(f"[bold red]Max steps ({self.max_steps}) reached.[/bold red]")

    async def run(
        self,
        state: StateT,
        artifact_service: BaseArtifactService | None = None,
    ) -> StateT:
        """Legacy/Convenience entry point.

        Wraps the ADK logic in a local execution loop.
        """
        # Create a mock Session and Context
        # We can't easily import 'Session' if it's not exposed,
        # but we can try to mimic it or use the one from adk_wrapper if available.

        import uuid

        from google.adk.sessions import InMemorySessionService, Session

        # Initialize session with the Pydantic state dumped as dict
        session = Session(
            id="local_run",
            app_name="markov-agent",
            user_id="test-user",
            state=state.model_dump(),
        )

        service_to_use = artifact_service or InMemoryArtifactService()

        context = InvocationContext(
            session=session,
            session_service=InMemorySessionService(),
            invocation_id=str(uuid.uuid4()),
            agent=self,
            artifact_service=service_to_use,
        )

        # Run the generator
        async for _ in self._run_async_impl(context):
            pass  # We just consume events, the work happens in session.state

        # Update the original state object with results from session
        return state.update(**session.state)

    async def run_beam(
        self,
        initial_state: StateT,
        width: int = 3,
        max_steps: int = 10,
    ) -> list[StateT]:
        """Execute the graph using Beam Search to find the most probable paths."""
        import copy
        import uuid

        from google.adk.artifacts import InMemoryArtifactService
        from google.adk.sessions import InMemorySessionService, Session

        # candidates stores (state, current_node_id)
        candidates: list[tuple[StateT, str]] = [(initial_state, self.entry_point)]
        final_states: list[StateT] = []

        for _ in range(max_steps):
            next_candidates: list[tuple[StateT, str]] = []

            for state, node_id in candidates:
                if node_id not in self.nodes:
                    final_states.append(state)
                    continue

                node = self.nodes[node_id]

                # Execute Node on a clone of the state
                branch_state = copy.deepcopy(state)
                session = Session(
                    id=f"beam_{uuid.uuid4()}",
                    app_name=self.name,
                    user_id="beam-search",
                    state=branch_state.model_dump(),
                )
                ctx = InvocationContext(
                    session=session,
                    session_service=InMemorySessionService(),
                    invocation_id=str(uuid.uuid4()),
                    agent=self,
                    artifact_service=InMemoryArtifactService(),
                )

                # Run node
                async for _ in node._run_async_impl(ctx):
                    pass

                # Update branch_state from session safely
                branch_state = type(state).model_validate(copy.deepcopy(session.state))

                # Find transitions
                found_transition = False
                for edge in self.edges:
                    if edge.source == node_id:
                        distribution = edge.get_distribution(branch_state)
                        if not distribution:
                            continue

                        found_transition = True
                        for next_node_id, prob in distribution.items():
                            child_state = copy.deepcopy(branch_state)
                            if hasattr(child_state, "record_probability"):
                                child_state.record_probability(
                                    node_id, prob, distribution=distribution
                                )
                            next_candidates.append((child_state, next_node_id))
                        break

                if not found_transition:
                    final_states.append(branch_state)

            if not next_candidates:
                candidates = []
                break

            # Sort by confidence and prune
            next_candidates.sort(
                key=lambda x: x[0].meta.get("confidence", 1.0), reverse=True
            )
            candidates = next_candidates[:width]

        # Combine results
        all_results = final_states + [c[0] for c in candidates]
        all_results.sort(key=lambda x: x.meta.get("confidence", 1.0), reverse=True)
        return all_results[:width]
