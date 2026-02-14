import asyncio
import copy
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from markov_agent.core.state import BaseState
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class ParallelNode(BaseNode[StateT]):
    """Executes a list of nodes in parallel.

    Merges the resulting state updates.
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
        self._schema_fields = {}
        if self.state_type and hasattr(self.state_type, "model_fields"):
            self._schema_fields = self.state_type.model_fields

    def _get_merged_updates(
        self,
        initial_state: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge updates from multiple branches respecting 'append' behavior."""
        merged_updates: dict[str, Any] = {}

        for res_dump in results:
            for key, value in res_dump.items():
                old_val = initial_state.get(key)
                if value == old_val:
                    continue

                # Check behavior
                is_append = False
                if key in self._schema_fields:
                    field = self._schema_fields[key]
                    if (
                        field.json_schema_extra
                        and field.json_schema_extra.get("behavior") == "append"
                    ):
                        is_append = True

                if is_append and isinstance(value, list) and isinstance(old_val, list):
                    # Append logic: calculate new items
                    new_items = value[len(old_val) :]
                    if not new_items:
                        continue

                    if key in merged_updates:
                        merged_updates[key].extend(new_items)
                    else:
                        merged_updates[key] = list(new_items)
                else:
                    # Last write wins
                    merged_updates[key] = value

        return merged_updates

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Execute sub-nodes in parallel with state isolation."""
        from google.adk.sessions import Session

        # 1. Snapshot State
        initial_state = copy.deepcopy(ctx.session.state)

        async def run_node_isolated(
            node: BaseNode,
        ) -> tuple[list[Event], dict[str, Any]]:
            isolated_session = Session(
                id=f"{ctx.session.id}_{node.name}",
                app_name=ctx.session.app_name,
                user_id=ctx.session.user_id,
                state=copy.deepcopy(initial_state),
            )

            isolated_context = InvocationContext(
                session=isolated_session,
                session_service=ctx.session_service,
                invocation_id=ctx.invocation_id,
                agent=node,
            )

            events = [event async for event in node._run_async_impl(isolated_context)]

            return events, isolated_session.state

        # 2. Run in parallel
        results = await asyncio.gather(
            *(run_node_isolated(node) for node in self.nodes),
        )

        # 3. Process Results and Merge
        final_states = []
        for events, final_node_state in results:
            for event in events:
                yield event
            final_states.append(final_node_state)

        merged_updates = self._get_merged_updates(dict(initial_state), final_states)

        # 4. Update Main State respecting append behavior for session state
        for key, value in merged_updates.items():
            is_append = False
            if key in self._schema_fields:
                field = self._schema_fields[key]
                if (
                    field.json_schema_extra
                    and field.json_schema_extra.get("behavior") == "append"
                ):
                    is_append = True

            if is_append and key in ctx.session.state and isinstance(value, list):
                # Manually append to the session state list
                if isinstance(ctx.session.state[key], list):
                    ctx.session.state[key].extend(value)
                else:
                    ctx.session.state[key] = value
            else:
                ctx.session.state[key] = value

    async def execute(self, state: StateT) -> StateT:
        """Run all sub-nodes in parallel using the initial state."""
        # 1. Execute all nodes concurrently
        results = await asyncio.gather(*(node.execute(state) for node in self.nodes))

        # 2. Merge strategies
        initial_dump = state.model_dump()
        result_dumps = [res.model_dump() for res in results]

        merged_updates = self._get_merged_updates(initial_dump, result_dumps)

        # 3. Create new state with merged updates
        return state.update(**merged_updates)
