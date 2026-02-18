import json

import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.token_utils import ReductionStrategy


@pytest.mark.asyncio
async def test_holistic_state_reduction_llm():
    """Test that ADKController correctly uses LLM for holistic state reduction."""

    reduction_calls = []

    def mock_responder(prompt: str) -> str:
        if "REDUCE THIS STATE" in prompt:
            reduction_calls.append(prompt)
            # Simulate LLM returning a valid reduced JSON state
            return json.dumps(
                {"summary": "State was reduced by LLM", "critical_val": 42}
            )
        return "Final response"

    config = ADKConfig(
        model_name="mock-model",
        max_input_tokens=100,  # Very low budget
        reduction_strategy=ReductionStrategy.LLM,
        reduction_prompt="REDUCE THIS STATE",
        mock_responder=mock_responder,
    )

    controller = ADKController(config=config, retry_policy=RetryPolicy())

    # Large state that will exceed the budget
    large_state = {
        "key1": "Large data " * 50,
        "key2": "Even more data " * 50,
        "key3": [1, 2, 3] * 50,
    }

    prompt = "Some prompt"
    _, final_state = await controller._reduce_context(
        prompt, large_state, max_tokens=100
    )

    # Verify reduction was called
    assert len(reduction_calls) == 1
    assert "REDUCE THIS STATE" in reduction_calls[0]
    assert "key1" in reduction_calls[0]

    # Verify final state is the one returned by the reduction LLM
    assert final_state == {"summary": "State was reduced by LLM", "critical_val": 42}


@pytest.mark.asyncio
async def test_holistic_state_reduction_llm_fallback():
    """Test that LLM reduction falls back to standard reduction if LLM fails."""

    def mock_responder(prompt: str) -> str:
        if "REDUCE THIS STATE" in prompt:
            err_msg = "LLM Reduction Failed"
            raise RuntimeError(err_msg)
        return "Final response"

    config = ADKConfig(
        model_name="mock-model",
        max_input_tokens=50,
        reduction_strategy=ReductionStrategy.LLM,
        reduction_prompt="REDUCE THIS STATE",
        mock_responder=mock_responder,
    )

    controller = ADKController(config=config, retry_policy=RetryPolicy())

    large_state = {"key1": "A" * 500, "key2": "B" * 500}

    prompt = "Some prompt"
    # This should not crash, but fallback to standard (greedy) reduction
    _, final_state = await controller._reduce_context(
        prompt, large_state, max_tokens=50
    )

    # Verify it was reduced (some keys should be truncated)
    assert final_state is not None
    assert len(json.dumps(final_state)) < 500
    assert "[TRUNCATED]" in (final_state.get("key1", "") or "") or "[TRUNCATED]" in (
        final_state.get("key2", "") or ""
    )
