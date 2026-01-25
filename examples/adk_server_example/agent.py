import logging
from collections.abc import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from markov_agent.core.state import BaseState
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode


# 1. Define State
class SimpleState(BaseState):
    input_text: str = ""
    output_text: str = ""


# 2. Define Node
class EchoNode(BaseNode[SimpleState]):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Access state directly from session (dict)
        state_dict = ctx.session.state
        input_text = state_dict.get("input_text", "")

        response = f"Echo: {input_text}"

        # Update State
        ctx.session.state["output_text"] = response

        # Yield Event with Content (so API Server shows it)
        yield Event(
            author=self.name,
            actions=EventActions(),
            content=types.Content(role="model", parts=[types.Part(text=response)]),
        )

    # We can still keep execute for local usage if we want, but _run_async_impl is primary for ADK
    async def execute(self, state: SimpleState) -> SimpleState:
        response = f"Echo: {state.input_text}"
        return state.update(output_text=response)


# 3. Define Topology
node = EchoNode(name="echo_node", state_type=SimpleState)

# Edge: Always terminal for this simple example
edges = []

# 4. Create Graph (The Agent)
# This 'agent' variable is what ADK API Server looks for
agent = Graph(
    name="simple_server_agent",
    nodes={"echo_node": node},
    edges=edges,
    entry_point="echo_node",
    state_type=SimpleState,
    input_key="input_text",  # Maps user message to state.input_text
)

# Optional: Add logging
logging.basicConfig(level=logging.INFO)
