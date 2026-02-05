import os
from unittest.mock import patch

import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.topology.edge import Edge
from markov_agent.topology.graph import Graph
from markov_agent.topology.node import BaseNode

# --- Graph Max Steps Test ---


class CountNode(BaseNode):
    async def execute(self, state):
        # Increment a counter in the state
        # The state is untyped/generic in this test context unless we define it

        # Handle StateProxy behavior where getattr returns None for missing keys
        current = getattr(state, "steps", 0)
        if current is None:
            current = 0

        state.steps = current + 1
        return state


@pytest.mark.asyncio
async def test_graph_max_steps_termination():
    """Test that an infinite loop (A->B->A) terminates at exactly max_steps."""
    node_a = CountNode(name="A")
    node_b = CountNode(name="B")

    # Cyclic edges
    edge_ab = Edge(source="A", target_func=lambda s: "B")
    edge_ba = Edge(source="B", target_func=lambda s: "A")

    graph = Graph(
        name="infinite_loop",
        nodes={"A": node_a, "B": node_b},
        edges=[edge_ab, edge_ba],
        entry_point="A",
        max_steps=5,
    )

    class CountState(BaseState):
        steps: int = 0

    final_state = await graph.run(CountState())

    # It should run exactly 5 times (A, B, A, B, A) -> Stop
    # Or A(0), B(1), A(2), B(3), A(4) -> 5 steps?
    # Graph implementation: while steps < max_steps.
    # Starts at 0. Runs node. steps+=1.
    # 0: Run A. steps=1.
    # 1: Run B. steps=2.
    # 2: Run A. steps=3.
    # 3: Run B. steps=4.
    # 4: Run A. steps=5.
    # Loop check: 5 < 5 is False. Break.
    # So 5 executions.

    assert final_state.steps == 5


# --- LiteLLM Integration Test ---


def test_litellm_initialization():
    """Verify that ADKController initializes LiteLLM when use_litellm=True."""
    config = ADKConfig(
        model_name="openai/gpt-4o",
        use_litellm=True,
        api_base="http://localhost:1234",
        api_key="sk-test",
    )
    retry = RetryPolicy()

    # Mock google.adk.models.lite_llm.LiteLlm
    # And os.environ

    with patch.dict(os.environ, {}, clear=True):
        with patch("google.adk.models.lite_llm.LiteLlm") as MockLiteLlm:
            # Mock Agent to avoid real init
            with patch("markov_agent.engine.adk_wrapper.Agent"):
                with patch("markov_agent.engine.adk_wrapper.Runner"):
                    with patch("markov_agent.engine.adk_wrapper.App"):
                        ADKController(config, retry)

            MockLiteLlm.assert_called_once()
            _, _kwargs = MockLiteLlm.call_args


# --- State Proxy Test ---


@pytest.mark.asyncio
async def test_graph_state_proxy_fallback():
    """Verify that if state_type is None, the graph creates a proxy
    that allows attribute access.
    """
    node_a = CountNode(name="A")

    # Edge function accessing state attribute
    def check_attr(s):
        if s.special_flag == "yes":
            return None  # Terminate
        return "A"  # Loop forever if not handled (but we want termination)

    edge = Edge(source="A", target_func=check_attr)

    graph = Graph(
        name="proxy_test",
        nodes={"A": node_a},
        edges=[edge],
        entry_point="A",
        state_type=None,  # Important!
    )

    # We need to manually inject a dict state that has 'special_flag'
    # Graph.run() expects a BaseState usually.
    # We must pass a BaseState to .run(), but Graph internally won't know the type
    # because state_type=None.

    class DynamicState(BaseState):
        special_flag: str

    initial = DynamicState(special_flag="yes")

    # The graph will treat it as generic dict inside ctx.session.state.
    # Then _run_async_impl creates StateProxy(ctx.session.state).
    # Then edge function calls s.special_flag.

    await graph.run(initial)

    # If no error raised, proxy worked.
