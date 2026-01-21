from typing import TypeVar

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class NestedGraphNode(BaseNode[StateT]):
    """
    Wraps a Graph instance to be used as a Node within another topology.
    Allows for recursive/nested graph structures.
    """

    def __init__(
        self, name: str, graph: Graph, state_type: type[StateT] | None = None, **kwargs
    ):
        super().__init__(name=name, state_type=state_type, **kwargs)
        self.graph = graph

    async def _run_async_impl(self, context: Any) -> Any:
        """
        Delegates execution to the wrapped Graph's ADK implementation.
        """
        async for event in self.graph._run_async_impl(context):
            yield event

    async def execute(self, state: StateT) -> StateT:
        """
        Delegates execution to the wrapped Graph.
        """
        # We can emit an event here if we want to trace nesting
        return await self.graph.run(state)
