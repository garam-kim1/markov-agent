import pytest
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import StreamingMode

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.runtime import RunConfig


def test_run_config_streaming_modes():
    # Test default
    rc = RunConfig()
    adk_rc = rc.to_adk_run_config()
    assert adk_rc.streaming_mode == StreamingMode.NONE

    # Test legacy streaming flag
    rc = RunConfig(streaming=True)
    adk_rc = rc.to_adk_run_config()
    assert adk_rc.streaming_mode == StreamingMode.SSE

    # Test explicit sse
    rc = RunConfig(streaming_mode="sse")
    adk_rc = rc.to_adk_run_config()
    assert adk_rc.streaming_mode == StreamingMode.SSE

    # Test bidi
    rc = RunConfig(streaming_mode="bidi")
    adk_rc = rc.to_adk_run_config()
    assert adk_rc.streaming_mode == StreamingMode.BIDI


def test_run_config_modalities():
    rc = RunConfig(response_modalities=["AUDIO", "TEXT"])
    adk_rc = rc.to_adk_run_config()
    assert adk_rc.response_modalities == ["AUDIO", "TEXT"]


@pytest.mark.asyncio
async def test_adk_controller_run_live_delegation():
    from unittest.mock import MagicMock

    from google.adk.events import Event

    config = ADKConfig(model_name="mock-model")
    controller = ADKController(config=config, retry_policy=RetryPolicy())

    # Mock the runner's run_live method
    mock_event = MagicMock(spec=Event)

    async def mock_run_live_gen(*args, **kwargs):
        yield mock_event

    controller.runner.run_live = MagicMock(side_effect=mock_run_live_gen)  # type: ignore

    queue = MagicMock(spec=LiveRequestQueue)
    run_config = RunConfig(streaming_mode="bidi")

    events = [
        event
        async for event in controller.run_live(
            live_request_queue=queue,
            run_config=run_config,
            session_id="test_session",
        )
    ]

    assert len(events) == 1
    assert events[0] == mock_event
    controller.runner.run_live.assert_called_once()
    # Check that it was called with the correct queue and run_config
    call_kwargs = controller.runner.run_live.call_args.kwargs
    assert call_kwargs["live_request_queue"] == queue
    assert call_kwargs["run_config"].streaming_mode == StreamingMode.BIDI
