from abc import ABC, abstractmethod
from typing import AsyncGenerator, TypeVar, Any, Generic

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from pydantic import ConfigDict

from markov_agent.core.state import BaseState

StateT = TypeVar("StateT", bound=BaseState)


class BaseNode(BaseAgent, Generic[StateT], ABC):
    """
    Abstract Base Node, wrapping google.adk.agents.BaseAgent.
    Integrates Markov Agent's Pydantic State with ADK's session state.
    """
    
    # Allow arbitrary types for Pydantic (needed for ADK internals if any)
    # Allow extra fields so subclasses can set attributes in __init__ without declaring fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    def __init__(self, name: str, description: str = "", **kwargs):
        # BaseAgent expects name, description, sub_agents (list), prompt_config, etc.
        # We pass minimal args here and let subclasses handle specifics
        super().__init__(name=name, description=description, **kwargs)

    async def execute(self, state: StateT) -> StateT:
        """
        Public API for Markov Agent users.
        Bridges the Pydantic State to ADK's InvocationContext/Session.
        """
        # 1. Create a dummy InvocationContext/Session for the execution
        # (In a real full ADK app, this comes from the Runner)
        # For 'wrapper' usage, we might need to mock this or use InMemorySessionService
        # But wait, if we are just a wrapper, maybe we shouldn't define 'execute' like this?
        # We should encourage using the ADK Runner.
        # However, to maintain backward compatibility with the 'markov-agent' style:
        
        # We'll use a local shim to run the agent's logic
        # This is a bit hacky but allows 'node.execute(state)' to work
        # by calling '_run_async_impl' directly or similar.
        
        # NOTE: Properly mocking the context is complex. 
        # For now, we will assume this method is used for unit testing or simple execution
        # where we manually invoke the logic.
        
        # Ideally: Create context -> run_async -> collect events -> return updated state
        pass
        
    # We must implement _run_async_impl from BaseAgent (or run_async_impl per SDK)
    # Python SDK usually uses _run_async_impl
    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        The ADK execution entry point.
        Subclasses should implement their logic here or in a specialized method.
        If using the Markov Agent wrapper pattern, we bridge to 'execute'.
        """
        # Bridge to 'execute' for legacy/wrapper support
        
        # 1. Reconstruct State from session (Best effort)
        # We need to know the State class. We can't know it easily here without Generic inspection
        # or assuming the user kept 'state' as a dict if they didn't provide a Pydantic model.
        # But 'execute' expects StateT.
        
        # HACK: We assume the user creates nodes that handle the state they are given.
        # If 'execute' expects a Pydantic model, we attempt to convert.
        # But we don't have the class ref.
        
        # We will try to pass a proxy object that mimics the Pydantic model
        class StateProxy:
            def __init__(self, data):
                self.__dict__ = data
            def __getattr__(self, name):
                return self.__dict__.get(name)
            def update(self, **kwargs):
                # Mimic Pydantic update: return NEW dict (or proxy)
                new_data = self.__dict__.copy()
                new_data.update(kwargs)
                return StateProxy(new_data)
            def model_dump(self):
                return self.__dict__
        
        # Use proxy or raw dict depending on strictness
        # For our AppendNode test, it does `getattr(state, field)`. Proxy works.
        # It does `state.update`. Proxy works.
        
        state_input = StateProxy(context.session.state)
        
        # 2. Call execute
        try:
            new_state = await self.execute(state_input)
        except Exception:
            # If execute fails (maybe type check), we might yield error
            # But for now let it raise to debug
            raise

        # 3. Update Session State
        if new_state:
            # Merge updates back
            if hasattr(new_state, 'model_dump'):
                context.session.state.update(new_state.model_dump())
            elif isinstance(new_state, dict):
                context.session.state.update(new_state)
            elif isinstance(new_state, StateProxy):
                context.session.state.update(new_state.__dict__)

        # Default behavior: generic event
        yield Event(
            author=self.name,
            actions=EventActions()
        )


