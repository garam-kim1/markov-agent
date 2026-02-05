from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class NestedGraphNode(BaseNode[StateT]):
    """Wraps a Graph instance to be used as a Node within another topology.

    Allows for recursive/nested graph structures.
    """

    def __init__(
        self,
        name: str,
        graph: Graph,
        state_type: type[StateT] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, state_type=state_type, **kwargs)
        self.graph = graph

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Delegate execution to the wrapped Graph's ADK implementation."""
        async for event in self.graph._run_async_impl(ctx):  # noqa: SLF001
            yield event

    async def execute(self, state: StateT) -> StateT:
        """Delegate execution to the wrapped Graph."""
        # We can emit an event here if we want to trace nesting
        return await self.graph.run(state)
