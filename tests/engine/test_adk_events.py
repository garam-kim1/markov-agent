from unittest.mock import AsyncMock, MagicMock

import pytest

from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy


@pytest.fixture
def mock_adk_components():
    config = ADKConfig(model_name="gemini-1.5-flash")
    controller = ADKController(config=config, retry_policy=RetryPolicy(max_attempts=1))

    # Mock the runner and session service
    controller.runner = MagicMock()
    controller.session_service = AsyncMock()

    return controller


@pytest.mark.asyncio
async def test_run_async_yields_events(mock_adk_components):
    controller = mock_adk_components

    # Setup mock event stream
    mock_event = MagicMock()
    mock_event.is_final_response.return_value = True

    async def mock_run_async(*args, **kwargs):
        yield mock_event

    controller.runner.run_async.side_effect = mock_run_async

    events = [event async for event in controller.run_async(prompt="test")]

    assert len(events) == 1
    assert events[0] == mock_event
    controller.session_service.create_session.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_events(mock_adk_components):
    controller = mock_adk_components

    mock_session = MagicMock()
    mock_event = MagicMock()
    mock_session.events = [mock_event]
    controller.session_service.get_session.return_value = mock_session

    events = await controller.get_session_events(session_id="test_session")

    assert len(events) == 1
    assert events[0] == mock_event
    controller.session_service.get_session.assert_called_once_with(
        app_name="markov_agent",
        user_id="system",
        session_id="test_session",
    )
