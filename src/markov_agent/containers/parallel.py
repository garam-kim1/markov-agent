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
        Executes sub-nodes in parallel with state isolation.
        We snapshot the state, run each node in its own isolated context,
        and then merge the updates back to the main session.
        """
        import asyncio
        import copy

        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.sessions import Session

        # 1. Snapshot State
        initial_state = copy.deepcopy(context.session.state)

        async def run_node_isolated(node):
            # Create isolated session/context
            # Note: We share session_service but use a distinct session ID logic or just a transient session object
            isolated_session = Session(
                id=f"{context.session.id}_{node.name}",
                appName=context.session.appName,
                userId=context.session.userId,
                state=copy.deepcopy(initial_state),
            )

            isolated_context = InvocationContext(
                session=isolated_session,
                session_service=context.session_service,
                invocation_id=context.invocation_id,
                agent=node,
            )

            events = []
            async for event in node._run_async_impl(isolated_context):
                events.append(event)

            return events, isolated_session.state

        # 2. Run in parallel
        results = await asyncio.gather(
            *(run_node_isolated(node) for node in self.nodes)
        )

        # 3. Process Results and Merge
        merged_updates = {}
        for events, final_node_state in results:
            # Yield events
            for event in events:
                yield event

            # Identify changes
            for key, value in final_node_state.items():
                if value != initial_state.get(key):
                    # Last writer wins logic for conflicts, but with parallel branches usually touching different keys
                    # In a real conflict, we might need a better strategy, but this mimics 'execute' logic.
                    merged_updates[key] = value

        # 4. Update Main State
        context.session.state.update(merged_updates)

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
