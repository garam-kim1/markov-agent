from collections.abc import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from markov_agent import BaseNode, BaseState, Graph


# 1. Define State
class DeploymentState(BaseState):
    input_text: str = ""
    output_text: str = ""


# 2. Define Node
class SimpleEchoNode(BaseNode[DeploymentState]):
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # Access state directly from session (dict)
        state_dict = ctx.session.state
        input_text = state_dict.get("input_text", "")

        response = f"Deployed Echo: {input_text}"

        # Update State
        ctx.session.state["output_text"] = response

        # Yield Event with Content (so API Server shows it)
        yield Event(
            author=self.name,
            actions=EventActions(),
            content=types.Content(role="model", parts=[types.Part(text=response)]),
        )


# 3. Define Topology
node = SimpleEchoNode(name="echo_node", state_type=DeploymentState)

# 4. Create Graph (The Agent)
# This 'agent' variable is exported for main.py
agent = Graph(
    name="deployment_example_agent",
    nodes={"echo_node": node},
    edges=[],  # Single node graph
    entry_point="echo_node",
    state_type=DeploymentState,
    input_key="input_text",
)
