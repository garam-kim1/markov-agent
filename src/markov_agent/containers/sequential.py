from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class SequentialNode(BaseNode[StateT]):
    """Executes a list of nodes sequentially.

    Lighter weight alternative to creating a full Chain Graph.
    """

    def __init__(
        self,
        name: str,
        nodes: list[BaseNode],
        state_type: type[StateT] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, state_type=state_type)
        self.nodes = nodes

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Execute sub-nodes sequentially."""
        for node in self.nodes:
            # We assume node is a BaseNode (Agent)
            async for event in node._run_async_impl(ctx):  # noqa: SLF001
                yield event

    async def execute(self, state: StateT) -> StateT:
        for node in self.nodes:
            state = await node.execute(state)
        return state
