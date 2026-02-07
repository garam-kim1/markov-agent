import pytest

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode


class SimpleState(BaseState):
    input: str
    thought: str = ""
    output: str = ""


@pytest.mark.asyncio
async def test_probabilistic_node_deep_reasoning():
    """Test the 'deep' reasoning path in ProbabilisticNode."""

    responses = {
        "Think about {{ input }}": "I should say hello",
        "Based on your thoughts: I should say hello, answer the user: {{ input }}": "Hello there!",
    }

    def mock_resp(prompt):
        for p in responses:
            # Simple matching for mock
            if (
                p.replace("{{ input }}", "test") in prompt
                or "I should say hello" in prompt
            ):
                if "Think about" in prompt:
                    return "I should say hello"
                return "Hello there!"
        return "Default response"

    config = ADKConfig(model_name="mock")
    node = ProbabilisticNode(
        name="reasoner",
        adk_config=config,
        prompt_template="Based on your thoughts: {{ thought }}, answer the user: {{ input }}",
        mock_responder=mock_resp,
    )

    # Custom deep implementation for test
    async def custom_deep(self, state):
        thinking_prompt = f"Think about {state.input}"
        # This calls the internal controller
        thought = await self.controller.generate(thinking_prompt)
        state = state.update(thought=thought)
        # Then run the normal execute
        return await self._execute_impl(state)

    # Monkeypatch deep for this node instance if needed,
    # but ProbabilisticNode has a 'deep' method we can use if we subclass.

    class DeepNode(ProbabilisticNode):
        async def execute(self, state):
            thinking_prompt = f"Think about {state.input}"
            thought = await self.controller.create_variant(
                {"response_mime_type": "text/plain"}
            ).generate(thinking_prompt)
            state = state.update(thought=thought)
            return await self._execute_impl(state)

    node = DeepNode(
        name="reasoner",
        adk_config=config,
        prompt_template="Based on your thoughts: {{ thought }}, answer the user: {{ input }}",
        mock_responder=mock_resp,
    )

    state = SimpleState(input="test")
    new_state = await node.execute(state)

    assert new_state.thought == "I should say hello"
    assert "Hello there!" in str(new_state.history[-1]["output"])


def test_probabilistic_node_render_prompt_with_dict():
    """Test _render_prompt when state is a dict and state_type is provided."""
    config = ADKConfig(model_name="mock")
    node = ProbabilisticNode(
        name="test",
        adk_config=config,
        prompt_template="Hello {{ input }}",
        state_type=SimpleState,
    )

    # Test with dict that can be validated
    prompt = node._render_prompt({"input": "world"})
    assert prompt == "Hello world"

    # Test with dict that might fail validation but should still work via construct/dict
    prompt = node._render_prompt({"input": "universe", "extra": "ignored"})
    assert prompt == "Hello universe"
