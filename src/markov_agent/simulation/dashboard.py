import asyncio
import json
import uuid
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import Session
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from markov_agent.core.services import ServiceRegistry
from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph

StateT = TypeVar("StateT", bound=BaseState)


class DashboardRunner:
    """Runs a Graph with a rich interactive dashboard for better UX/Observability."""

    def __init__(
        self,
        graph: Graph,
        initial_state: StateT,
        refresh_rate: int = 4,
        delay: float = 0.5,
    ) -> None:
        self.graph = graph
        self.state = initial_state
        self.refresh_rate = refresh_rate
        self.delay = delay
        self.console = Console()
        self.logs: list[str] = []
        self.current_node: str = graph.entry_point
        self.step_count: int = 0
        self.latest_event: str = ""

    def _create_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="state", ratio=2),
            Layout(name="metrics", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="graph", ratio=1),
            Layout(name="logs", ratio=2),
        )
        return layout

    def _update_view(self, layout: Layout, session_state: dict[str, Any]) -> None:
        # Header
        layout["header"].update(
            Panel(
                f"[bold blue]Markov Agent Dashboard[/bold blue] - {self.graph.name}",
                style="white on blue",
            )
        )

        # State (JSON View)
        # Filter out history/meta/logs for clean view
        clean_state = {
            k: v
            for k, v in session_state.items()
            if k not in ["history", "meta", "logs"]
        }
        json_str = json.dumps(clean_state, indent=2, default=str)
        layout["state"].update(
            Panel(
                Syntax(json_str, "json", theme="monokai", line_numbers=True),
                title="Current State",
                border_style="cyan",
            )
        )

        # Metrics
        metrics_table = Table(expand=True, box=None)
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Value", style="bold")
        metrics_table.add_row("Step", str(self.step_count))
        metrics_table.add_row("Current Node", self.current_node)

        # Add custom metrics from state if available
        if "budget" in session_state:
            val = session_state["budget"]
            metrics_table.add_row(
                "Budget", f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            )
        if "revenue" in session_state:
            val = session_state["revenue"]
            metrics_table.add_row(
                "Revenue", f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            )
        if "confidence" in session_state.get("meta", {}):
            metrics_table.add_row(
                "Confidence", f"{session_state['meta']['confidence']:.2f}"
            )

        layout["metrics"].update(
            Panel(metrics_table, title="Metrics", border_style="green")
        )

        # Graph / Node Info
        node_info = Text()
        node_info.append(f"Active Node: {self.current_node}\\n", style="bold yellow")
        if self.current_node in self.graph.nodes:
            node = self.graph.nodes[self.current_node]
            node_info.append(f"Type: {type(node).__name__}\\n", style="dim")
            if hasattr(node, "prompt_template") and node.prompt_template:
                preview = node.prompt_template[:100].replace("\\n", " ") + "..."
                node_info.append(f"Prompt: {preview}\\n", style="italic dim")

        layout["graph"].update(
            Panel(node_info, title="Topology Status", border_style="yellow")
        )

        # Logs
        log_text = Text()
        # Mix session logs if available
        session_logs = session_state.get("logs", [])
        if not session_logs:
            meta = session_state.get("meta", {})
            if isinstance(meta, dict):
                session_logs = meta.get("logs", [])
        display_logs = (self.logs + session_logs)[-20:]

        for log in display_logs:
            style = "white"
            if "Error" in log:
                style = "red"
            elif "Success" in log:
                style = "green"
            elif "[" in log and "]" in log:
                style = "cyan"
            log_text.append(f"{log}\\n", style=style)

        layout["logs"].update(Panel(log_text, title="Event Log", border_style="white"))

        # Footer
        layout["footer"].update(
            Panel(
                f"Status: Running... | Latest Event: {self.latest_event[:80]}...",
                style="dim",
            )
        )

    async def run(self) -> StateT:
        """Execute the graph with live dashboard."""
        # Setup Session context (copied from Graph.run logic)
        session = Session(
            id=f"dash_{uuid.uuid4()}",
            app_name="markov-agent-dashboard",
            user_id="user",
            state=self.state.model_dump(),
        )

        context = InvocationContext(
            session=session,
            session_service=ServiceRegistry.get_session_service(),
            invocation_id=str(uuid.uuid4()),
            agent=self.graph,
            artifact_service=ServiceRegistry.get_artifact_service(),
        )

        layout = self._create_layout()

        with Live(layout, refresh_per_second=self.refresh_rate, screen=True) as live:
            self.logs.append("Simulation started.")

            # Hook into the generator from Graph._run_async_impl
            # We must call _run_async_impl on the graph instance
            async for event in self.graph._run_async_impl(context):
                # Update internal tracking
                if event.author:
                    self.current_node = event.author

                content = ""
                if event.content and event.content.parts:
                    content = event.content.parts[0].text or ""
                    self.latest_event = content.replace("\\n", " ")
                    # Try to parse node output for logging
                    if len(content) < 100:
                        self.logs.append(f"[{event.author}] {content}")
                    else:
                        self.logs.append(
                            f"[{event.author}] Output generated ({len(content)} chars)"
                        )

                self.step_count += 1

                # Update UI
                self._update_view(layout, session.state)
                live.refresh()

                # Delay for readability
                if self.delay > 0:
                    await asyncio.sleep(self.delay)

            self.logs.append("Simulation complete.")
            self._update_view(layout, session.state)
            live.refresh()
            # Keep final state visible for a moment
            await asyncio.sleep(2.0)

        # Reconstruct typed state
        return type(self.state).model_validate(session.state)
