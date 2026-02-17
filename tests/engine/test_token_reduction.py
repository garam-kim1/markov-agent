import pytest

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.agent import VerifiedAgent
from markov_agent.engine.token_utils import ReductionStrategy


@pytest.mark.asyncio
async def test_token_reduction_verified_agent():
    # Large research result that should be reduced
    large_research = "This is a very long research result. " * 1000

    # Mock responder that returns the large research
    def mock_responder(prompt: str) -> str:
        if "Research" in prompt:
            return large_research
        # Return something to signal if research was reduced
        if "[TRUNCATED]" in prompt:
            return "Detected truncated research"
        return "Normal implementation"

    agent = VerifiedAgent(
        name="test_reducer",
        model="gemini-1.5-flash",  # Use a recognized model name
        max_input_tokens=500,  # Set a low limit
        mock_responder=mock_responder,
    )

    # We want to check if the second call (implementation) received a truncated prompt
    # Since we can't easily peek into the internal call, we use the mock_responder
    # to return a specific string if it sees [TRUNCATED].

    result = await agent.run_verified("Do something")

    assert result == "Detected truncated research"


@pytest.mark.asyncio
async def test_token_reduction_direct_controller():
    from markov_agent.engine.adk_wrapper import ADKController, RetryPolicy

    config = ADKConfig(model_name="mock-model", max_input_tokens=100)

    received_prompt = ""

    def mock_responder(prompt: str) -> str:
        nonlocal received_prompt
        received_prompt = prompt
        return "ok"

    controller = ADKController(
        config=config, retry_policy=RetryPolicy(), mock_responder=mock_responder
    )

    large_prompt = "Large prompt " * 200
    await controller.generate(large_prompt)

    assert "[TRUNCATED]" in received_prompt
    # Ensure it's roughly within budget (100 tokens is small)
    from markov_agent.engine.token_utils import count_tokens

    assert count_tokens(received_prompt) <= 100


@pytest.mark.asyncio
async def test_llm_based_reduction():
    from markov_agent.engine.adk_wrapper import ADKController, RetryPolicy

    # We use a mock responder that handles both the reduction call and the final call
    def mock_responder(prompt: str) -> str:
        if "REDUCE THIS" in prompt:
            return "REDUCED_BY_LLM"
        return f"FINAL_RESPONSE_WITH_{prompt}"

    config = ADKConfig(
        model_name="mock-model",
        max_input_tokens=50,
        reduction_strategy=ReductionStrategy.LLM,
        reduction_prompt="REDUCE THIS",
    )

    controller = ADKController(
        config=config, retry_policy=RetryPolicy(), mock_responder=mock_responder
    )

    # Large prompt to trigger reduction
    large_prompt = "Very large prompt that needs reduction " * 50
    result = await controller.generate(large_prompt)

    # The final prompt should contain the reduced version
    assert "REDUCED_BY_LLM" in result


@pytest.mark.asyncio
async def test_importance_sampling_reduction():
    from markov_agent.engine.adk_wrapper import ADKController, RetryPolicy
    from markov_agent.engine.token_utils import ReductionStrategy

    received_prompt = ""

    def mock_responder(prompt: str) -> str:
        nonlocal received_prompt
        received_prompt = prompt
        return "ok"

    config = ADKConfig(
        model_name="mock-model",
        max_input_tokens=50,
        reduction_strategy=ReductionStrategy.IMPORTANCE,
        recency_weight=5.0,
    )

    controller = ADKController(
        config=config, retry_policy=RetryPolicy(), mock_responder=mock_responder
    )

    # A prompt with some rare words and repetitive words
    # Rare words: 'Zylophone', 'Quasar', 'Nebula'
    # Repetitive: 'the', 'is', 'a'
    prompt = (
        "the is a " * 20
        + "Zylophone "
        + "the is a " * 20
        + "Quasar "
        + "the is a " * 20
        + "Nebula"
    )

    await controller.generate(prompt)

    # Check if rare words are preserved
    assert "Zylophone" in received_prompt
    assert "Quasar" in received_prompt
    assert "Nebula" in received_prompt

    # Check that it's within token limit
    from markov_agent.engine.token_utils import count_tokens

    assert count_tokens(received_prompt) <= 50
    # Unlike greedy, importance sampling shouldn't have [TRUNCATED] unless we add it
    assert "[TRUNCATED]" not in received_prompt
