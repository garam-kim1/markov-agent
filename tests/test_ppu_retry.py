
import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


@pytest.mark.asyncio
async def test_retry_transient_failure():
    """Test that transient failures are retried and eventually succeed."""

    # Setup mock to fail 2 times then succeed
    attempts = 0

    def flaky_responder(prompt):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError(f"Fail attempt {attempts}")
        return "Success"

    retry = RetryPolicy(
        max_attempts=4,
        initial_delay=0.01,  # Fast for test
        backoff_factor=1.0,
    )
    config = ADKConfig(model_name="mock-model")

    controller = ADKController(config, retry, mock_responder=flaky_responder)

    result = await controller.generate("Test")
    assert result == "Success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_permanent_failure():
    """Test that permanent failures exhaust retries and raise RuntimeError."""

    def always_fail(prompt):
        raise ValueError("Permanent Fail")

    retry = RetryPolicy(max_attempts=3, initial_delay=0.01, backoff_factor=1.0)
    config = ADKConfig(model_name="mock-model")

    controller = ADKController(config, retry, mock_responder=always_fail)

    with pytest.raises(RuntimeError, match="Failed to generate after 3 attempts"):
        await controller.generate("Test")
