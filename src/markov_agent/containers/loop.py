from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar, cast

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

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
        state_type: type[StateT] | None = None,
        **kwargs,
    ):
        """
        Args:
            name: Node name.
            body: The node to execute in the loop.
            condition: A function that returns True if the loop should STOP.
            max_iterations: Maximum number of iterations.
            state_type: The Pydantic model class for the state.
        """
        super().__init__(name=name, state_type=state_type)
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Create a proxy for condition checking or use typed state
        class StateProxy:
            def __init__(self, data):
                self.__dict__ = data

            def __getattr__(self, name):
                return self.__dict__.get(name)

        for _ in range(self.max_iterations):
            # Check condition on current state
            state_obj: Any
            if self.state_type:
                try:
                    state_obj = self.state_type.model_validate(ctx.session.state)
                except Exception:
                    state_obj = self.state_type.construct(**ctx.session.state)
            else:
                state_obj = StateProxy(ctx.session.state)

            if self.condition(cast(StateT, state_obj)):
                break

            async for event in self.body._run_async_impl(ctx):
                yield event

    async def execute(self, state: StateT) -> StateT:
        for _ in range(self.max_iterations):
            if self.condition(state):
                break
            state = await self.body.execute(state)

            # Check immediately after execution as well to exit early?
            # No, let the loop structure handle it.

        return state
