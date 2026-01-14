import asyncio
from typing import TypeVar

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class ParallelNode(BaseNode[StateT]):
    """
    Executes a list of nodes in parallel.
    Merges the resulting state updates.
    """

    def __init__(self, name: str, nodes: list[BaseNode], **kwargs):
        super().__init__(name=name)
        self.nodes = nodes

    async def _run_async_impl(self, context: Any) -> Any:
        """
        Executes sub-nodes in parallel.
        Note: ADK's ParallelAgent handles this more robustly with branching.
        Here we do a simple gather and merge on the shared session state.
        This is risky for race conditions on the dictionary!
        """
        import asyncio
        
        # We need to capture initial state to merge later?
        # Or we let them fight over the dict (standard Python dict is thread-safe for single ops but not logic)
        # Since we are async, we are in one thread.
        # But if they read-then-write, we have race conditions.
        
        # For this wrapper, we'll run them sequentially? No, ParallelNode implies parallel.
        # We will run them, collecting their events.
        
        # We can't easily merge generator streams in a simple gathering without a helper.
        # We'll use a helper to consume a generator.
        
        async def run_node(node):
            events = []
            async for event in node._run_async_impl(context):
                events.append(event)
            return events

        results = await asyncio.gather(*(run_node(node) for node in self.nodes))
        
        for event_list in results:
            for event in event_list:
                yield event

    async def execute(self, state: StateT) -> StateT:
        """
        Runs all sub-nodes in parallel using the initial state.
        Then merges the changes from each branch back into the main state.
        """
        # 1. Execute all nodes concurrently
        results = await asyncio.gather(*(node.execute(state) for node in self.nodes))

        # 2. Merge strategies
        # Since State is immutable (Pydantic), we detect changes by comparing
        # the result state dictionaries with the initial state dictionary.
        initial_dump = state.model_dump()
        merged_updates = {}

        for res in results:
            res_dump = res.model_dump()
            for key, value in res_dump.items():
                # If value changed from initial, we want to keep it
                if value != initial_dump.get(key):
                    # Potential conflict check could go here
                    merged_updates[key] = value

        # 3. Create new state with merged updates
        return state.update(**merged_updates)
