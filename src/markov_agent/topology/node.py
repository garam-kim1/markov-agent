from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Self, TypeVar, cast

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import ConfigDict, Field

from markov_agent.core.state import BaseState

StateT = TypeVar("StateT", bound=BaseState)


class BaseNode[StateT](BaseAgent, ABC):
    """Abstract Base Node, wrapping google.adk.agents.BaseAgent.

    Integrates Markov Agent's Pydantic State with ADK's session state.
    """

    # Allow arbitrary types for Pydantic (needed for ADK internals if any)
    # Allow extra fields so subclasses can set attributes in __init__
    # without declaring fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    state_type: Any = Field(default=None, exclude=True)

    def __init__(
        self,
        name: str,
        description: str = "",
        state_type: type[StateT] | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseAgent expects name, description, sub_agents (list), prompt_config, etc.
        # We pass minimal args here and let subclasses handle specifics
        super().__init__(name=name, description=description, **kwargs)
        self.state_type = state_type

    @abstractmethod
    async def execute(self, state: StateT) -> StateT:
        """Public API for Markov Agent users.

        Bridges the Pydantic State to ADK's InvocationContext/Session.
        """

    def __rshift__(self, other: Any) -> Any:
        """Support operator overloading for flow definition."""
        from markov_agent.topology.edge import Edge, Flow

        if isinstance(other, (BaseNode, str)):
            target_name = other.name if hasattr(other, "name") else other
            new_edge = Edge(source=self.name, target=target_name)
            return Flow([new_edge], last_node=other)

        if isinstance(other, Edge):
            other.source = self.name
            # If edge has a target, we can return a Flow.
            # If it doesn't (like writer >> Edge(cond=...)), it will be completed by another >>
            if other.target:
                return Flow([other], last_node=other.target)
            return other

        msg = f"Cannot link {self.name} with {type(other)}"
        raise TypeError(msg)

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Execute the ADK execution entry point.

        Subclasses should implement their logic here or in a specialized method.
        If using the Markov Agent wrapper pattern, we bridge to 'execute'.
        """
        # Bridge to 'execute' for legacy/wrapper support

        # 1. Reconstruct State from session (Best effort)
        # We use explicit conversion if state_type is provided.

        if self.state_type:
            try:
                state_input = self.state_type.model_validate(ctx.session.state)
            except Exception:
                # Fallback if validation fails (e.g. partial state)
                # We might log a warning here in a real system
                state_input = self.state_type.construct(**ctx.session.state)
        else:
            # If no type provided, we assume the user's execute method handles dicts
            # or we use the legacy proxy approach if absolutely necessary,
            # but strictly we should pass the dict if no type is known.
            # However, existing nodes might expect dot-notation.
            class StateProxy:
                def __init__(self, data: dict[str, Any]) -> None:
                    self.__dict__ = data

                def __getattr__(self, name: str) -> Any:
                    return self.__dict__.get(name)

                def update(self, **kwargs: Any) -> Self:
                    new_data = self.__dict__.copy()
                    new_data.update(kwargs)
                    return StateProxy(new_data)  # type: ignore[return-value]

                def model_dump(self) -> dict[str, Any]:
                    return self.__dict__

            state_input = StateProxy(ctx.session.state)

        # 2. Call execute
        new_state = await self.execute(cast("StateT", state_input))

        # 3. Update Session State
        if new_state:
            # Merge updates back
            st: Any = new_state
            session_state = cast("dict[Any, Any]", ctx.session.state)
            if hasattr(st, "model_dump"):
                session_state.update(cast("Any", st.model_dump)())
            elif isinstance(st, dict):
                session_state.update(st)
            elif hasattr(st, "__dict__"):
                session_state.update(st.__dict__)

        # Default behavior: generic event
        yield Event(author=self.name, actions=EventActions())


class FunctionalNode[StateT](BaseNode[StateT]):
    """A node that executes a deterministic Python function."""

    def __init__(
        self,
        name: str,
        func: Callable[
            [StateT],
            StateT | dict[str, Any] | None | Awaitable[StateT | dict[str, Any] | None],
        ],
        description: str = "",
        state_type: type[StateT] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, description=description, state_type=state_type, **kwargs)
        self.func = func

    async def execute(self, state: StateT) -> StateT:
        import inspect

        if inspect.iscoroutinefunction(self.func):
            result = await cast("Awaitable[Any]", self.func(state))
        else:
            result = self.func(state)

        if result is None:
            return state

        if isinstance(result, BaseState):
            return cast("StateT", result)

        if isinstance(result, dict):
            # Use the update method of the state if available
            update_func = getattr(state, "update", None)
            if update_func and callable(update_func):
                return cast("StateT", update_func(**cast("dict[str, Any]", result)))

            # Fallback for dict-based states
            if hasattr(state, "model_dump"):
                state_dict = cast("Any", state).model_dump()
            else:
                state_dict = dict(cast("Any", state))

            state_dict.update(result)
            if self.state_type:
                return self.state_type.model_validate(state_dict)
            return cast("StateT", state_dict)

        return state
