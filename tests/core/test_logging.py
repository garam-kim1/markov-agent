import logging

import pytest

from markov_agent import FileLoggingPlugin, setup_llm_logging
from markov_agent.engine.plugins import CallbackContext, LlmRequest, LlmResponse


def test_setup_llm_logging(tmp_path):
    log_file = tmp_path / "test_llm.log"
    setup_llm_logging(log_file=str(log_file))

    # Trigger a log entry
    logger = logging.getLogger("google_adk.models.google_llm")
    logger.debug("Test log entry")

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test log entry" in content
    assert "google_adk.models.google_llm" in content


@pytest.mark.asyncio
async def test_file_logging_plugin(tmp_path):
    from unittest.mock import MagicMock

    log_file = tmp_path / "plugin.log"
    plugin = FileLoggingPlugin(log_file=str(log_file))

    ctx = MagicMock(spec=CallbackContext)
    ctx.agent_name = "TestAgent"

    req = MagicMock(spec=LlmRequest)
    part_req = MagicMock()
    part_req.text = "Hello"
    content_req = MagicMock()
    content_req.parts = [part_req]
    req.contents = [content_req]

    res = MagicMock(spec=LlmResponse)
    part_res = MagicMock()
    part_res.text = "World"
    content_res = MagicMock()
    content_res.parts = [part_res]
    res.content = content_res

    await plugin.before_model_callback(callback_context=ctx, llm_request=req)
    await plugin.after_model_callback(callback_context=ctx, llm_response=res)

    assert log_file.exists()
    content = log_file.read_text()
    assert "LLM REQUEST [TestAgent]: Hello" in content
    assert "LLM RESPONSE [TestAgent]: World" in content
