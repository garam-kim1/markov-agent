from collections.abc import AsyncGenerator
from typing import TypeVar

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import ConfigDict, Field

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge
from markov_agent.topology.node import BaseNode

try:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
except ImportError:

    class Console:
        def log(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            pass

    console = Console()

    def Panel(x, title=None):
        return x


StateT = TypeVar("StateT", bound=BaseState)


class Graph(BaseAgent):
    """
    The execution engine acting as a finite state machine, wrapped as an ADK Agent.
    """

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
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.nodes = nodes
        self.edges = edges
        self.entry_point = entry_point
        self.state_type = state_type
        self.input_key = input_key
        # Register sub_agents for ADK hierarchy if needed
        # self.sub_agents = list(nodes.values())

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Executes the graph topology within the ADK runtime.
        """
        # 0. Handle Input Injection (ADK API Server support)
        if ctx.user_content and ctx.user_content.parts:
            # Extract text from the first part
            input_text = ""
            for part in ctx.user_content.parts:
                if part.text:
                    input_text += part.text
            
            if input_text:
                console.log(f"Injecting input into state['{self.input_key}']: {input_text[:50]}...")
                ctx.session.state[self.input_key] = input_text

        current_node_id = self.entry_point
        steps = 0

        console.log(
            f"[bold green]Starting Graph Execution[/bold green] at {self.entry_point}"
        )

        # We need to ensure we are working with the latest state from the session
        # ctx.session.state is a MutableMapping (dict-like)

        while steps < self.max_steps:
            if current_node_id not in self.nodes:
                yield Event(author=self.name, actions=EventActions(escalate=True))
                break

            current_node = self.nodes[current_node_id]

            console.log(
                Panel(
                    f"Executing Node: [cyan]{current_node_id}[/cyan]", title="Step info"
                )
            )

            # Execute Node
            # We call the node's ADK implementation directly
            # This allows the node to read/write to ctx.session.state
            async for event in current_node._run_async_impl(ctx):
                yield event

            # Transition Logic
            # We construct the typed state object for the router if possible

            state_obj = ctx.session.state
            if self.state_type:
                try:
                    state_obj = self.state_type.model_validate(ctx.session.state)
                except Exception:
                    # Best effort construct
                    state_obj = self.state_type.construct(**ctx.session.state)
            else:
                # Fallback Proxy
                class StateProxy:
                    def __init__(self, data):
                        self.__dict__ = data

                    def __getattr__(self, name):
                        return self.__dict__.get(name)

                state_obj = StateProxy(ctx.session.state)

            # Find next node
            next_node_id = None
            for edge in self.edges:
                if edge.source == current_node_id:
                    try:
                        # Try passing the typed object
                        next_node_id = edge.target_func(state_obj)
                    except Exception:
                        # Fallback: pass the dict directly
                        next_node_id = edge.target_func(ctx.session.state)

                    console.log(f"Transition: {current_node_id} -> {next_node_id}")
                    break

            if next_node_id is None:
                console.log(
                    f"[bold yellow]Terminal node reached:[/bold yellow] "
                    f"{current_node_id}"
                )
                break

            current_node_id = next_node_id
            steps += 1

        if steps >= self.max_steps:
            console.log(f"[bold red]Max steps ({self.max_steps}) reached.[/bold red]")

    async def run(self, state: StateT) -> StateT:
        """
        Legacy/Convenience entry point.
        Wraps the ADK logic in a local execution loop.
        """
        # Create a mock Session and Context
        # We can't easily import 'Session' if it's not exposed, but we can try to mimic it
        # or use the one from adk_wrapper if available.

        # For this wrapper, we'll create a simple dict-holding class if ADK imports fail,
        # but since we inherit BaseAgent, we assume ADK is present.

        import uuid

        from google.adk.sessions import InMemorySessionService, Session

        # Initialize session with the Pydantic state dumped as dict
        session = Session(
            id="local_run",
            app_name="markov-agent",
            user_id="test-user",
            state=state.model_dump(),
        )

        context = InvocationContext(
            session=session,
            session_service=InMemorySessionService(),
            invocation_id=str(uuid.uuid4()),
            agent=self,
        )

        # Run the generator
        async for _ in self._run_async_impl(context):
            pass  # We just consume events, the work happens in session.state

        # Update the original state object with results from session
        return state.update(**session.state)
