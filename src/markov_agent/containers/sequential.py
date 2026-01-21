from typing import TypeVar

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class SequentialNode(BaseNode[StateT]):
    """
    Executes a list of nodes sequentially.
    Lighter weight alternative to creating a full Chain Graph.
    """

    def __init__(
        self,
        name: str,
        nodes: list[BaseNode],
        state_type: type[StateT] | None = None,
        **kwargs,
    ):
        super().__init__(name=name, state_type=state_type)
        self.nodes = nodes

    async def _run_async_impl(self, context: Any) -> Any:
        """
        Executes sub-nodes sequentially.
        """
        for node in self.nodes:
            # We assume node is a BaseNode (Agent)
            async for event in node._run_async_impl(context):
                yield event

    async def execute(self, state: StateT) -> StateT:
        for node in self.nodes:
            state = await node.execute(state)
        return state
