from collections.abc import Callable
from typing import Any

from markov_agent.core.state import BaseState
from markov_agent.topology.edge import Edge, Flow
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


class Workflow(Graph):
    """A simplified Graph designed for rapid pipeline and team construction.

    Workflows allow users to build complex graphs simply by providing an initial flow
    (using the >> operator) without needing to manually register nodes or define state types
    if they just want to pass dictionaries or strings between agents.
    """

    def __init__(
        self,
        name: str = "Workflow",
        flow: Flow | Edge | list[Edge] | None = None,
        state_type: type[BaseState] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, state_type=state_type, **kwargs)
        if flow:
            self.add_flow(flow)

    def add_flow(self, flow: Flow | Edge | list[Edge]) -> None:
        """Connect a flow and use its starting node as the entry point if none is set."""
        self.connect(flow)

        # If entry point is not set, try to infer it from the first edge
        if not self.entry_point:
            if isinstance(flow, Edge):
                if flow.source and flow.source in self.nodes:
                    self.entry_point = flow.source
            elif isinstance(flow, (list, Flow)) and len(flow) > 0:
                first_edge = flow[0]
                if first_edge.source and first_edge.source in self.nodes:
                    self.entry_point = first_edge.source

    @classmethod
    def from_chain(
        cls,
        nodes: list[str | BaseNode | Callable],
        name: str = "ChainedWorkflow",
        state_type: type[BaseState] | None = None,
        **kwargs: Any,
    ) -> "Workflow":
        """Quickly create a linear workflow from a list of nodes."""
        workflow = cls(name=name, state_type=state_type, **kwargs)
        workflow.chain(nodes)

        # Set entry point
        for node in nodes:
            if isinstance(node, str):
                workflow.entry_point = node
                break
            if isinstance(node, BaseNode):
                workflow.entry_point = node.name
                break
            if callable(node):
                func_name = getattr(node, "__name__", f"func_{id(node)}")
                if func_name == "<lambda>":
                    func_name = f"lambda_{id(node)}"
                workflow.entry_point = func_name
                break

        return workflow
