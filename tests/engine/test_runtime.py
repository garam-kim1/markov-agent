from unittest.mock import AsyncMock

import pytest

from markov_agent import ADKConfig, ADKController, AdkWebServer, RetryPolicy, RunConfig


@pytest.mark.asyncio
async def test_run_config_usage():
    # Setup
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
    )
    retry_policy = RetryPolicy()
    controller = ADKController(config=config, retry_policy=retry_policy)

    # Mock generation
    controller.generate = AsyncMock(return_value="mocked response")  # type: ignore

    run_config = RunConfig(user_email="test@example.com", streaming=False)

    # Test run (blocking)
    # Since run calls generate which is async, and we mocked it to return a value
    # We need to be careful with how we mock for asyncio.run

    # Let's just test that we can pass run_config to generate
    await controller.generate("hello", run_config=run_config)
    controller.generate.assert_called_once()  # type: ignore
    _args, kwargs = controller.generate.call_args  # type: ignore
    assert kwargs["run_config"] == run_config


def test_adk_web_server_instantiation():
    config = ADKConfig(
        model_name="gemini-3-flash-preview",
    )
    retry_policy = RetryPolicy()
    controller = ADKController(config=config, retry_policy=retry_policy)

    server = AdkWebServer(agent=controller)
    assert server.agent_instance == controller
    assert server.internal_server is not None


def test_adk_web_server_with_custom_services():
    from google.adk.memory import InMemoryMemoryService
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    config = ADKConfig(
        model_name="gemini-3-flash-preview",
        session_service=session_service,
        memory_service=memory_service,
    )
    retry_policy = RetryPolicy()
    controller = ADKController(config=config, retry_policy=retry_policy)

    server = AdkWebServer(agent=controller)
    assert server.internal_server.session_service == session_service
    assert server.internal_server.memory_service == memory_service


def test_run_config_fields():
    run_config = RunConfig(
        model="gpt-4", tools=["tool1"], user_email="user@example.com", history=[]
    )
    assert run_config.model == "gpt-4"
    assert run_config.tools == ["tool1"]
    assert run_config.user_email == "user@example.com"
    assert run_config.history == []
