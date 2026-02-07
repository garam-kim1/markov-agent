import json
from unittest.mock import MagicMock

import pytest
from google.adk.agents.callback_context import CallbackContext

from markov_agent.engine.trajectory import TrajectoryRecorderPlugin


@pytest.mark.asyncio
async def test_trajectory_recorder_delta(tmp_path):
    log_file = tmp_path / "traj.jsonl"
    plugin = TrajectoryRecorderPlugin(log_path=str(log_file))

    # Mock CallbackContext
    ctx = MagicMock(spec=CallbackContext)
    mock_session = MagicMock()
    mock_session.state = {"a": 1, "b": 2}
    ctx.session = mock_session
    ctx.agent_name = "node1"
    ctx.invocation_id = "inv1"

    # Before run
    await plugin.before_agent_callback(ctx)

    # Change state
    mock_session.state["a"] = 10
    mock_session.state["c"] = 3

    # After run
    await plugin.after_agent_callback(ctx)

    # Check log
    log = json.loads(log_file.read_text())

    assert log["agent"] == "node1"
    assert log["delta_s"] == {"a": 10, "c": 3}
    assert log["invocation_id"] == "inv1"
