from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from markov_agent.core.events import Event, event_bus
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


class Graph(BaseModel):
    """
    The execution engine acting as a finite state machine.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: dict[str, BaseNode]
    edges: list[Edge]
    entry_point: str
    max_steps: int = 50

    async def run(self, state: StateT) -> StateT:
        """
        Executes the graph topology.
        """
        current_node_id = self.entry_point
        steps = 0

        console.log(
            f"[bold green]Starting Graph Execution[/bold green] at {self.entry_point}"
        )
        await event_bus.emit(
            Event(name="graph_start", payload={"entry_point": self.entry_point})
        )

        while steps < self.max_steps:
            if current_node_id not in self.nodes:
                console.log(
                    f"[bold red]Error:[/bold red] Node {current_node_id} not found."
                )
                await event_bus.emit(
                    Event(
                        name="graph_error",
                        payload={"error": f"Node {current_node_id} not found"},
                    )
                )
                break

            current_node = self.nodes[current_node_id]
            console.log(
                Panel(
                    f"Executing Node: [cyan]{current_node_id}[/cyan]", title="Step info"
                )
            )

            # Execute Node Logic
            state = await current_node.execute(state)
            await event_bus.emit(
                Event(
                    name="node_executed",
                    payload={"node": current_node_id, "state": state.model_dump()},
                )
            )

            # Find next node via Edge
            next_node_id = None
            for edge in self.edges:
                if edge.source == current_node_id:
                    next_node_id = edge.route(state)
                    console.log(f"Transition: {current_node_id} -> {next_node_id}")
                    break

            if next_node_id is None:
                console.log(
                    f"[bold yellow]Terminal node reached:[/bold yellow] "
                    f"{current_node_id}"
                )
                await event_bus.emit(
                    Event(name="graph_end", payload={"exit_node": current_node_id})
                )
                break

            current_node_id = next_node_id
            steps += 1

        if steps >= self.max_steps:
            console.log(
                f"[bold red]Max steps ({self.max_steps}) reached.[/bold red] Halting."
            )
            await event_bus.emit(Event(name="graph_halted", payload={"steps": steps}))

        return state
