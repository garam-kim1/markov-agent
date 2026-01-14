from collections.abc import Callable
from typing import TypeVar

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class LoopNode(BaseNode[StateT]):
    """
    Executes a node (or chain of nodes) repeatedly until a condition is met
    or max_iterations is reached.
    """

    def __init__(
        self,
        name: str,
        body: BaseNode,
        condition: Callable[[StateT], bool],
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Args:
            name: Node name.
            body: The node to execute in the loop.
            condition: A function that returns True if the loop should STOP.
            max_iterations: Maximum number of iterations.
        """
        super().__init__(name=name)
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations

    async def _run_async_impl(self, context: Any) -> Any:
        # Create a proxy for condition checking
        class StateProxy:
            def __init__(self, data):
                self.__dict__ = data
            def __getattr__(self, name):
                return self.__dict__.get(name)

        for _ in range(self.max_iterations):
            # Check condition on current state
            proxy = StateProxy(context.session.state)
            if self.condition(proxy):
                break
                
            async for event in self.body._run_async_impl(context):
                yield event

    async def execute(self, state: StateT) -> StateT:
        for _ in range(self.max_iterations):
            if self.condition(state):
                break
            state = await self.body.execute(state)
            
            # Check immediately after execution as well to exit early?
            # No, let the loop structure handle it.
            
        return state
